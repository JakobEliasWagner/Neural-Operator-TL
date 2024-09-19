import argparse
import pathlib
from types import ModuleType
from collections import defaultdict
import csv
import pandas as pd
from scipy.stats import bootstrap
import continuiti
import notl  # noqa: F401
import torch
import torch.utils
import torch.utils.data
import yaml
from nos.data import TLDatasetCompact
from continuiti.transforms import Normalize
import numpy as np


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
        default="data/updated_test.csv",
        required=False,
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/updated.csv",
        required=False,
    )

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

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    train_dataset = TLDatasetCompact(pathlib.Path(args.train_csv), v_transform="normalize")
    dataset = TLDatasetCompact(pathlib.Path(args.test_csv))
    dataset.transform = train_dataset.transform
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # operator
    multirun_dir = pathlib.Path(args.run_dir)

    # different architectures
    for operator_dir in multirun_dir.iterdir():
        if not operator_dir.is_dir():
            # multi-run yaml
            continue

        best_path = operator_dir.joinpath("best")

        ys = []
        vs = []
        outs = []
        for run_dir in best_path.iterdir():
            run_ys = []
            run_vs = []
            run_outs = []
            operator = notl.load_run_operator(run_dir=run_dir, dataset_shapes=dataset.shapes, run_id=run_dir.name)

            operator.eval()
            with torch.no_grad():
                for x, u, y, v in dataloader:
                    out = operator(x, u, y)

                    v_u, out_u = dataset.transform["v"].undo(v), dataset.transform["v"].undo(out)
                    y_u = dataset.transform["y"].undo(y)

                    u_u = dataset.transform["u"].undo(u)

                    run_ys.append(y_u)
                    run_vs.append(v_u)
                    run_outs.append(out_u)
                    break
            ys.append(torch.cat(run_ys, dim=0))  # cat on batch dimension
            vs.append(torch.cat(run_vs, dim=0))  # cat on batch dimension
            outs.append(torch.cat(run_outs, dim=0))  # cat on batch dimension
        ys_t = torch.stack(ys)
        vs_t = torch.stack(vs)
        outs_t = torch.stack(outs)

        # mean
        mean = torch.mean(outs_t, dim=0, keepdim=True)

        # bootstrapped CI
        ci = bootstrap((outs_t.squeeze().numpy(),), np.mean, axis=0)

        df = pd.DataFrame.from_dict({
            "CI95Low": ci.confidence_interval.low.tolist(),
            "CI95High": ci.confidence_interval.high.tolist(),
            "std": ci.standard_error.squeeze().tolist(),
            "mean": mean.squeeze().tolist(),
            "y": ys_t[0, 0, 0, :].tolist(),
            "v": vs_t[0, 0, 0, :].tolist(),
        })
        df.to_csv(out_dir.joinpath(f"tl_{operator.__class__.__name__}.csv"), index=False)


if __name__ == "__main__":
    main()
