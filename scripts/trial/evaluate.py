import argparse
import pathlib
from types import ModuleType
import time
import continuiti
import notl  # noqa: F401
import torch
import torch.utils
import torch.utils.data
import yaml
from nos.data import TLDatasetCompact
from continuiti.operators import Operator
from continuiti.transforms import Normalize
from scipy.stats import bootstrap
import numpy as np


class UnknownOperatorModule(Exception):
    def __init__(self, module_name):
        super().__init__(f"Unknown operator module {module_name}.")


def proc_time(bs: int, es: int, model: Operator, n_iter: int = 100, ):
    model.eval()

    x = torch.rand(bs, 3, 1, device=torch.device("cuda"))
    u = torch.rand(bs, 3, 1, device=torch.device("cuda"))
    y = torch.rand(bs, 1, es, device=torch.device("cuda"))

    start = time.time()
    with torch.inference_mode(), torch.no_grad():
        for _ in range(n_iter):
            model(x, u, y)
    delta_t = time.time() - start
    return bs * n_iter * es / delta_t


def find_max_batch_size(model: Operator, evaluations: int = 256, initial_batch_size: int = 32,
                        max_iterations: int = 1024):
    """
    Finds the maximum batch size that fits into GPU memory without causing a CUDA out of memory error.

    Parameters:
    model (Operator): The neural operator model to test.
    evaluations (int): The number of evaluations in one observation.
    initial_batch_size (int): The starting batch size to test.
    max_iterations (int): The maximum number of iterations for the binary search.

    Returns:
    int: The maximum batch size that fits into GPU memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def can_allocate(bts: int):
        try:
            proc_time(bts, evaluations, model, n_iter=1)
            return True
        except RuntimeError:
            return False

    low = initial_batch_size
    high = None
    found_batch_size = low

    for _ in range(max_iterations):
        if high is None:
            batch_size = low * 2
        else:
            batch_size = (low + high) // 2

        if can_allocate(batch_size):
            found_batch_size = batch_size
            low = batch_size + 1
        else:
            high = batch_size - 1

        if high is not None and low > high:
            break

    torch.cuda.empty_cache()
    return found_batch_size


def get_args() -> argparse.Namespace:
    """Parse Arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str)
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

    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            operator.to(device)
            operator.eval()
            with torch.no_grad():
                for x, u, y, v in dataloader:
                    x, u, y = x.to(device), u.to(device), y.to(device)
                    out = operator(x, u, y)

                    v_u, out_u = dataset.transform["v"].undo(v), dataset.transform["v"].undo(out.detach().cpu())
                    y_u = dataset.transform["y"].undo(y.detach().cpu())

                    run_ys.append(y_u)
                    run_vs.append(v_u)
                    run_outs.append(out_u)
            ys.append(torch.cat(run_ys, dim=0))  # cat on batch dimension
            vs.append(torch.cat(run_vs, dim=0))  # cat on batch dimension
            outs.append(torch.cat(run_outs, dim=0))  # cat on batch dimension
        vs_t = torch.stack(vs)
        outs_t = torch.stack(outs)

        squared_error = (vs_t - outs_t) ** 2
        relative_squared_error = squared_error / torch.mean(vs_t ** 2)

        run_rmse = torch.mean(relative_squared_error.flatten(1, -1), dim=1)
        ci = bootstrap((run_rmse.numpy(),), np.mean)

        print("-" * 10, operator.__class__.__name__, "-" * 10)
        print(sum([p.numel() for p in operator.parameters() if p.requires_grad]))

        print("MEAN:\t", torch.mean(relative_squared_error).item())
        print("STD:\t", torch.std(relative_squared_error).item())
        print("ci:\t", ci.confidence_interval)

        """bs = find_max_batch_size(operator, evaluations=256, initial_batch_size=2**14, max_iterations=2 ** 9)
        print(f"Bs: {bs}")

        pt = proc_time(bs, 256, operator, n_iter=10)
        print("pt:\t", pt)
        print("Speedup:\t", pt/11.9)"""


if __name__ == "__main__":
    main()
