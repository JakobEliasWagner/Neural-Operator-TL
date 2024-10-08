import pathlib

import hydra
import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split


@hydra.main(version_base=None, config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:  # noqa: D103
    # config
    output_path = pathlib.Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger.add(output_path.joinpath("main.log"))
    with output_path.joinpath("config.yaml").open("w") as config_file:
        OmegaConf.save(cfg, config_file)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}.")

    # multiple runs with reproducible seeds
    for run_id in range(cfg.training.start_seed, cfg.training.end_seed):
        logger.info("-" * 10, f"Starting run {run_id}", "-" * 10)

        seed = run_id  # reproducible
        torch.manual_seed(seed)

        # dataset
        dataset = hydra.utils.instantiate(cfg.dataset, v_transform="normalize")
        train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])
        logger.info(f"Loaded dataset from {cfg.dataset.path} with {dataset.x.size(0)} observations.")

        # operator
        operator = hydra.utils.instantiate(cfg.operator.architecture, shapes=dataset.shapes).to(device)
        logger.info(
            f"Initialized {operator.__class__.__name__} with "
            f"{sum(p.numel() for p in operator.parameters() if p.requires_grad)} trainable parameters.",
        )

        # optimizer
        lr = 1e-3
        optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=cfg.training.epochs,
            eta_min=lr * 1e-2,  # reduce to 1% of initial
        )

        criterion = torch.nn.MSELoss()

        # save
        logger.info(f"Saving checkpoints to {output_path}.")

        best_dir = output_path.joinpath("best", f"run_{run_id}")
        best_dir.mkdir(exist_ok=True, parents=True)
        last_dir = output_path.joinpath("last", f"run_{run_id}")
        last_dir.mkdir(exist_ok=True, parents=True)

        # training
        batch_size = cfg.operator.hyperparameter.batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = torch.inf
        run_name = f"{operator.__class__.__name__}_{sum(p.numel() for p in operator.parameters() if p.requires_grad)}"
        logger.info(f"Starting training for {cfg.training.epochs} epochs with name {run_name}.")
        with mlflow.start_run(run_name=run_name):
            for epoch in range(cfg.training.epochs):
                train_loss = []
                operator.train()
                for x, u, y, v in train_loader:
                    x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
                    out = operator(x, u, y)

                    optimizer.zero_grad()
                    loss = criterion(v, out)
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                avg_train_loss = torch.mean(torch.tensor(train_loss))

                operator.eval()
                val_loss = []
                val_loss_u = []
                with torch.no_grad():
                    for x, u, y, v in val_loader:
                        x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
                        out = operator(x, u, y)

                        out_u, v_u = dataset.transform["v"].undo(out.detach().cpu()), dataset.transform["v"].undo(
                            v.detach().cpu())

                        loss = criterion(v, out)
                        loss_u = criterion(v_u, out_u)
                        val_loss.append(loss.item())
                        val_loss_u.append(loss_u.item())

                avg_val_loss = torch.mean(torch.tensor(val_loss)).item()
                avg_val_loss_u = torch.mean(torch.tensor(val_loss_u)).item()

                mlflow.log_metric("Train Loss", avg_train_loss.item(), step=epoch)
                mlflow.log_metric("Val Loss", avg_val_loss, step=epoch)
                mlflow.log_metric("Val Loss Unscaled", avg_val_loss_u, step=epoch)
                mlflow.log_metric("LR", optimizer.param_groups[0]["lr"], step=epoch)

                if avg_val_loss < best_val_loss:
                    torch.save(operator.state_dict(), best_dir.joinpath("operator.pt"))
                    best_val_loss = avg_val_loss

                scheduler.step()

            torch.save(operator.state_dict(), last_dir.joinpath("operator.pt"))
            logger.info(f"Finished training. The best mean validation loss is: {best_val_loss}.")


if __name__ == "__main__":
    main()
