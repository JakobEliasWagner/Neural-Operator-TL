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
from tqdm import tqdm
from nos.data import TLDatasetCompact
from continuiti.operators import Operator
from continuiti.transforms import Normalize


class UnknownOperatorModule(Exception):
    def __init__(self, module_name):
        super().__init__(f"Unknown operator module {module_name}.")


def proc_time(bs: int, es: int, model: Operator, n_iter: int = 100, ):
    model.eval()
    torch.cuda.synchronize()

    x = torch.rand(bs, 3, 1, device=torch.device("cuda"))
    u = torch.rand(bs, 3, 1, device=torch.device("cuda"))
    y = torch.rand(bs, 1, es, device=torch.device("cuda"))

    start = time.time()
    with torch.inference_mode():
        for _ in tqdm(range(n_iter)):
            model(x, u, y)
    delta_t = time.time() - start
    torch.cuda.synchronize()
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
            with torch.inference_mode():
                x = torch.rand(bts, 3, 1, device=torch.device("cuda"))
                u = torch.rand(bts, 3, 1, device=torch.device("cuda"))
                y = torch.rand(bts, 1, evaluations, device=torch.device("cuda"))
                model(x, u, y)
            return True
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                return False
            else:
                raise e

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

    return found_batch_size


def get_args() -> argparse.Namespace:
    """Parse Arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_dataset = TLDatasetCompact(pathlib.Path(args.train_csv))
    v_mean = torch.mean(train_dataset.v)
    v_std = torch.std(train_dataset.v)
    train_dataset.transform["v"] = Normalize(v_mean.reshape(1, 1), v_std.reshape(1, 1))
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
        operator_conf_path = operator_dir.joinpath("config.yaml")  # one specific configuration
        with operator_conf_path.open("r") as file:
            operator_conf = yaml.safe_load(file)
        operator_target = operator_conf["operator"]["architecture"]["_target_"].split(".")
        package = operator_target[0]
        class_name = operator_target[-1]

        if class_name != "DeepNeuralOperator":
            continue

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

            operator.to(device)
            operator = torch.compile(operator)
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
        mean = torch.mean(squared_error).item()
        std = torch.std(squared_error).item()
        print("-" * 10, class_name, "-" * 10)
        print(sum([p.numel() for p in operator.parameters() if p.requires_grad]))
        print("mean:\t", mean)
        print("std:\t", std)

        bs = find_max_batch_size(operator, evaluations=256, initial_batch_size=32, max_iterations=2 ** 9)
        print(f"Bs: {bs}")

        pt = proc_time(bs, 256, operator, n_iter=10)
        print("pt:\t", pt)


if __name__ == "__main__":
    main()
