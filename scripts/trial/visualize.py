import argparse
import pathlib
from types import ModuleType

import continuiti
import matplotlib.pyplot as plt
import notl  # noqa: F401
import torch
import torch.utils
import torch.utils.data
import yaml
from nos.data import TLDatasetCompact


class UnknownOperatorModule(Exception):
    def __init__(self, module_name):
        super().__init__(f"Unknown operator module {module_name}.")


def get_args() -> argparse.Namespace:
    """Parse Arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str)
    parser.add_argument("--out-dir", type=str, default="out", required=False)
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/test_smooth.csv",
        required=False,
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/smooth.csv",
        required=False,
    )
    parser.add_argument("--n-plots", type=int, default=2, required=False)

    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    """_summary_.

    _extended_summary_

    Args:
    ----
        args (_type_, optional): _description_. Defaults to None.

    """
    if args is None:
        args = get_args()

    # dataset
    train_dataset = TLDatasetCompact(pathlib.Path(args.train_csv), n_samples=args.n_plots)
    dataset = TLDatasetCompact(pathlib.Path(args.test_csv), n_samples=args.n_plots)
    dataset.transform = train_dataset.transform
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # operator
    multirun_dir = pathlib.Path(args.run_dir)

    # different architectures
    for operator_dir in multirun_dir.iterdir():
        if not operator_dir.is_dir():
            # multi-run yaml
            continue
        operator_conf_path = operator_dir.joinpath("config.yaml")  # one specific configuration
        with operator_conf_path.open("r") as file:
            operator_conf = yaml.safe_load(file)
        operator_target = operator_conf["operator"]["architecture"]["_target_"].split(".")
        package = operator_target[0]
        class_name = operator_target[-1]

        operator_module: ModuleType
        if package == "continuiti":
            operator_module = __import__("continuiti.operators", fromlist=["co"])
        elif package == "notl":
            operator_module = __import__("notl")
        else:
            raise UnknownOperatorModule(package)
        operator_class = getattr(operator_module, class_name)
        operator_args = {k: v for k, v in operator_conf["operator"]["architecture"].items() if k != "_target_"}

        best_path = operator_dir.joinpath("best")

        ys = []
        vs = []
        outs = []
        for run_dir in best_path.iterdir():
            run_ys = []
            run_vs = []
            run_outs = []
            operator: continuiti.operators.Operator = operator_class(dataset.shapes, **operator_args)
            operator.load_state_dict(torch.load(run_dir.joinpath("operator.pt")))

            operator.eval()
            with torch.no_grad():
                for x, u, y, v in dataloader:
                    out = operator(x, u, y)

                    v_u, out_u = dataset.transform["v"].undo(v), dataset.transform["v"].undo(out)
                    y_u = dataset.transform["y"].undo(y)

                    run_ys.append(y_u)
                    run_vs.append(v_u)
                    run_outs.append(out_u)
            ys.append(torch.cat(run_ys, dim=0))  # cat on batch dimension
            vs.append(torch.cat(run_vs, dim=0))  # cat on batch dimension
            outs.append(torch.cat(run_outs, dim=0))  # cat on batch dimension
        ys_t = torch.stack(ys)
        vs_t = torch.stack(vs)
        outs_t = torch.stack(outs)

        squared_error = (outs_t - vs_t) ** 2
        squared_error = torch.mean(squared_error.flatten(-2, -1), dim=-1)

        for tst_sample_id in range(squared_error.size(1)):
            median, median_id = torch.median(squared_error, dim=0)

            fig, ax = plt.subplots()

            ax.plot(outs_t[median_id, tst_sample_id, 0, :], ys_t[median_id, tst_sample_id, 0, :] / 1e3, "k-")
            ax.plot(vs_t[median_id, tst_sample_id, 0, :], ys_t[median_id, tst_sample_id, 0, :] / 1e3, "g--")

            ax.set_xlim(-100, 10)
            ax.set_ylim(2, 20)

            fig.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
