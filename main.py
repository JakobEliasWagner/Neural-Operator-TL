import pathlib
from torch.utils.data import random_split, DataLoader
import torch
from nos.data import TLDatasetCompact
from continuiti.operators import DeepNeuralOperator
import mlflow


def main() -> None:
    # conf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # objects
    dataset = TLDatasetCompact(pathlib.Path('data/2024-08-19_16-36-39_transmission_loss_dataset.csv'))
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    # operator
    operator = DeepNeuralOperator(shapes=dataset.shapes, width=64, depth=16).to(device)

    # optimizer
    lr = 1e-3
    optimizer = torch.optim.Adam(operator.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()

    # training

    train_loader, val_loader = DataLoader(train_dataset, batch_size=16), DataLoader(val_dataset, batch_size=16)

    with mlflow.start_run():
        for epoch in range(1000):
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
            with torch.no_grad():
                for x, u, y, v in val_loader:
                    x, u, y, v = x.to(device), u.to(device), y.to(device), v.to(device)
                    out = operator(x, u, y)

                    loss = criterion(v, out)
                    val_loss.append(loss.item())
            avg_val_loss = torch.mean(torch.tensor(val_loss))

            mlflow.log_metric('Train Loss', avg_train_loss.item(), step=epoch)
            mlflow.log_metric('Val Loss', avg_val_loss.item(), step=epoch)
            mlflow.log_metric('LR', optimizer.param_groups[0]['lr'], step=epoch)


if __name__ == "__main__":
    main()
