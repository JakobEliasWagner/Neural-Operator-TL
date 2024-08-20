import datetime
import pathlib

import mlflow
import torch
from continuiti.operators import DeepNeuralOperator
from loguru import logger
from nos.data import TLDatasetCompact
from torch.utils.data import DataLoader, random_split


def main(epochs: int = 500, lr: float = 1e-3) -> None:
    """Train neural operator play ground."""
    # conf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Running on {device}.")

    # objects
    # path = pathlib.Path("data/2024-08-19_16-36-39_transmission_loss_dataset.csv")
    path = pathlib.Path("data/smooth.csv")

    dataset = TLDatasetCompact(path, n_samples=-1)
    logger.info(f"Successfully loaded dataset from {path}.")
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    # operator
    operator = DeepNeuralOperator(shapes=dataset.shapes, width=32, depth=16).to(device)
    logger.info(f"Initialized {operator.__class__.__name__}.")
    logger.info(f"Model has {sum(p.numel() for p in operator.parameters() if p.requires_grad)} trainable parameters.")

    # optimizer
    lr = 1e-3
    optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs, eta_min=lr * 1e-2)

    criterion = torch.nn.MSELoss()

    # save
    model_checkpoint_dir = pathlib.Path("out")

    current_time = datetime.datetime.now()  # noqa: DTZ005
    time_stamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    model_stamp = f"{time_stamp}_{operator.__class__.__name__}"
    run_dir = model_checkpoint_dir.joinpath(model_stamp)

    best_dir = run_dir.joinpath("best")
    best_dir.mkdir(exist_ok=True, parents=True)
    last_dir = run_dir.joinpath("last")
    last_dir.mkdir(exist_ok=True, parents=True)

    # training
    train_loader, val_loader = DataLoader(train_dataset, batch_size=16), DataLoader(val_dataset, batch_size=16)
    best_val_loss = torch.inf

    logger.info(f"Starting training for {epochs} epochs.")
    with mlflow.start_run(run_name=model_stamp):
        for epoch in range(epochs):
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

                    out_u, v_u = dataset.transform["v"].undo(out), dataset.transform["v"].undo(v)

                    loss = criterion(v, out)
                    loss_u = criterion(v_u, out_u)
                    val_loss.append(loss.item())
                    val_loss_u.append(loss_u.item())
            scheduler.step()

            avg_val_loss = torch.mean(torch.tensor(val_loss)).item()
            avg_val_loss_u = torch.mean(torch.tensor(val_loss_u)).item()

            mlflow.log_metric("Train Loss", avg_train_loss.item(), step=epoch)
            mlflow.log_metric("Val Loss", avg_val_loss, step=epoch)
            mlflow.log_metric("Val Loss Unscaled", avg_val_loss_u, step=epoch)
            mlflow.log_metric("LR", optimizer.param_groups[0]["lr"], step=epoch)

            if avg_val_loss < best_val_loss:
                logger.info(f"Saving new best operator with val loss {avg_val_loss:.2E} in epoch {epoch}.")
                torch.save(operator, best_dir.joinpath("operator.pt"))
                best_val_loss = avg_val_loss
        torch.save(operator, last_dir.joinpath("operator.pt"))
        logger.info(f"Finished training for {epochs} epochs.")
        logger.info(f"Best validation loss: {best_val_loss:.2E}.")


if __name__ == "__main__":
    main()
