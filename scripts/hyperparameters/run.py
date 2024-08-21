import hydra
import optuna
import torch
from omegaconf import DictConfig
from optuna.storages import RetryFailedTrialCallback
from torch.utils.data import DataLoader, random_split

EPOCHS = 500


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    # conf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    def objective(trial) -> None:  # noqa: ANN001
        """_summary_.

        _extended_summary_

        Args:
        ----
            trial (_type_): _description_

        Raises:
        ------
            optuna.exceptions.TrialPruned: _description_

        """
        op_param = {name: trial.suggest_int(name, val[0], val[1]) for name, val in cfg.operator.parameter_space.items()}
        # operator
        operator = hydra.utils.instantiate(cfg.operator.architecture, shapes=dataset.shapes, **op_param).to(device)

        # optimizer
        lr = trial.suggest_float("lr", cfg.training.lr[0], cfg.training.lr[1], log=True)
        optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=EPOCHS, eta_min=lr * 1e-2)

        criterion = torch.nn.MSELoss()

        # training
        bs = trial.suggest_int("batch_size", cfg.training.batch_size[0], cfg.training.batch_size[1])

        train_loader, val_loader = DataLoader(train_dataset, batch_size=bs), DataLoader(val_dataset, batch_size=bs)

        for epoch in range(EPOCHS):
            operator.train()
            for x, u, y, v in train_loader:
                x_d, u_d, y_d, v_d = x.to(device), u.to(device), y.to(device), v.to(device)
                out = operator(x_d, u_d, y_d)

                optimizer.zero_grad()
                loss = criterion(v_d, out)
                loss.backward()
                optimizer.step()

            operator.eval()
            val_loss = []
            with torch.no_grad():
                for x, u, y, v in val_loader:
                    x_d, u_d, y_d, v_d = x.to(device), u.to(device), y.to(device), v.to(device)
                    out = operator(x_d, u_d, y_d)

                    loss = criterion(v_d, out)
                    val_loss.append(loss.item())
            avg_val_loss = torch.mean(torch.tensor(val_loss)).item()

            scheduler.step()

            # optuna reporting & pruning
            trial.report(avg_val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned

        return avg_val_loss

    storage = optuna.storages.RDBStorage(
        "sqlite:///hyperparameters.db",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage,
        study_name=f"{cfg.operator.architecture._target_}",
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=50),
    )
    study.optimize(objective, n_trials=30)


if __name__ == "__main__":
    main()
