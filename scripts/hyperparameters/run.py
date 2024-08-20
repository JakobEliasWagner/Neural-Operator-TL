from omegaconf import DictConfig
import hydra
from torch.utils.data import random_split, DataLoader
import optuna
from optuna.storages import RetryFailedTrialCallback
import torch


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    # conf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    def objective(trial):
        op_param = {
            name: trial.suggest_int(name, val[0], val[1]) for name, val in cfg.operator.parameter_space.items()
        }
        # operator
        operator = hydra.utils.instantiate(cfg.operator.architecture, shapes=dataset.shapes, **op_param).to(device)

        # optimizer
        lr = trial.suggest_float("lr", cfg.training.lr[0], cfg.training.lr[1], log=True)
        optimizer = torch.optim.Adam(operator.parameters(), lr=lr)

        criterion = torch.nn.MSELoss()

        # training
        epochs = trial.suggest_int("epochs", cfg.training.epochs[0], cfg.training.epochs[1])
        bs = trial.suggest_int("batch_size", cfg.training.batch_size[0], cfg.training.batch_size[1])

        train_loader, val_loader = DataLoader(train_dataset, batch_size=bs), DataLoader(val_dataset, batch_size=bs)

        for epoch in range(epochs):

            operator.train()
            for x, u, y, v in train_loader:
                x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
                out = operator(x, u, y)

                optimizer.zero_grad()
                loss = criterion(v, out)
                loss.backward()
                optimizer.step()

            operator.eval()
            val_loss = []
            with torch.no_grad():
                for x, u, y, v in val_loader:
                    x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
                    out = operator(x, u, y)

                    loss = criterion(v, out)
                    val_loss.append(loss.item())
            avg_val_loss = torch.mean(torch.tensor(val_loss))

            # optuna reporting & pruning
            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage,
        study_name=f"{cfg.operator.architecture._target_}",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )
    study.optimize(objective, n_trials=30)


if __name__ == "__main__":
    main()
