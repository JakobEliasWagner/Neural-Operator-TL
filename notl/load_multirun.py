import pathlib  # noqa: D100
from typing import TYPE_CHECKING

import torch
import yaml
from continuiti.operators import Operator, OperatorShapes

if TYPE_CHECKING:
    from types import ModuleType


class UnknownOperatorModuleError(Exception):
    """Unknown Operator Module Error.

    Args:
        Exception (_type_): _description_

    """

    def __init__(self, module_name: str) -> None:  # noqa: D107
        super().__init__(f"Unknown operator module {module_name}.")


def load_run_operator(run_dir: pathlib.Path, dataset_shapes: OperatorShapes, run_id: str | None = None) -> Operator:
    """Load an operator for a run dir.

    Args:
        run_dir: Run directory containing the config yaml and best dir for one specific operator.
        dataset_shapes: Shape of the dataset used for initialization.
        run_id: String Identifying run withing best dir of the multirun. Defaults to "run_0".

    """
    # load run configuration
    operator_conf_path = run_dir.joinpath("config.yaml")
    operator_conf: dict
    with operator_conf_path.open("r") as file:
        operator_conf = yaml.safe_load(file)

    # extract information
    operator_target = operator_conf["operator"]["architecture"]["_target_"].split(".")
    package = operator_target[0]
    class_name = operator_target[-1]

    # find and load correct operator parent module
    operator_module: ModuleType
    if package == "continuiti":
        operator_module = __import__("continuiti.operators", fromlist=["co"])
    elif package == "notl":
        operator_module = __import__("notl")
    else:
        raise UnknownOperatorModuleError(package)
    operator_class = getattr(operator_module, class_name)
    operator_args = {k: v for k, v in operator_conf["operator"]["architecture"].items() if k != "_target_"}

    # define operator architecture
    operator: Operator = operator_class(dataset_shapes, **operator_args)

    # load operator checkpoint
    if run_id is None:
        run_id = "run_0"
    operator_path = run_dir.joinpath("best", run_id, "operator.pt")
    operator.load_state_dict(torch.load(operator_path, weights_only=False))

    return operator
