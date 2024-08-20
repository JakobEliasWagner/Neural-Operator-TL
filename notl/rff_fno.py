import rff
from math import prod
import torch
from continuiti.operators import FourierNeuralOperator, NeuralOperator, OperatorShapes
from torch import nn


class RFFFourierNeuralOperator(NeuralOperator):
    """_summary_.

    _extended_summary_

    Args:
    ----
        NeuralOperator (_type_): _description_

    """

    def __init__(
        self,
        shapes: OperatorShapes,
        depth: int = 3,
        width: int = 3,
        sigma: float = 1.,
        encoding_size: int = 32,
        act: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        """_summary_.

        _extended_summary_

        Args:
        ----
            shapes (OperatorShapes): _description_
            depth (int): _description_
            width (int): _description_
            sigma (float): _description_
            encoding_size (int): _description_
            act (torch.nn.Module, optional): _description_. Defaults to None.
            device (torch.device, optional): _description_. Defaults to None.

        """
        super().__init__(shapes, device)

        self.encoder = rff.Gaussian(sigma=sigma, input_size=prod(shapes.u.size), encoded_size=encoding_size)

        fno_shape = shapes.copy()
        fno_shape.u.size = encoding_size
        self.fno = FourierNeuralOperator(shapes=fno_shape, depth=depth, width=width, act=act, device=device)

    def forward(self, _: torch.Tensor, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """_summary_.

        _extended_summary_

        Args:
        ----
            _ (None): _description_
            u (torch.Tensor): _description_
            y (torch.Tensor): _description_

        Returns:
        -------
            torch.Tensor: _description_

        """
        encoding = self.encoder(u)

        return self.fno(_, encoding, y)
