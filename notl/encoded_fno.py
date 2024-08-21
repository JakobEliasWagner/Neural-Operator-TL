from abc import ABC, abstractmethod
from copy import deepcopy
from math import prod

import rff
import torch
from continuiti.operators import FourierNeuralOperator, Operator, OperatorShapes
from torch import nn


class Encoder(nn.Module, ABC):
    """_summary_.

    _extended_summary_

    Args:
    ----
        nn (_type_): _description_
        ABC (_type_): _description_

    """

    def __init__(self, encoding_size: int) -> None:
        """_summary_.

        _extended_summary_
        """
        super().__init__()
        self.encoding_size = encoding_size

    def get_linspace_domain(self, size: torch.Size) -> torch.Tensor:
        """_summary_.

        _extended_summary_

        Args:
        ----
            size (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        x = torch.linspace(-1, 1, size[-1])
        x = x.reshape(1, 1, -1)
        return x.expand(size[0], -1, -1)

    @abstractmethod
    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_.

        _extended_summary_

        Args:
        ----
            src (_type_): _description_

        Raises:
        ------
            ValueError: _description_

        Returns:
        -------
            _type_: _description_

        """


class RFFEncoder(Encoder):
    """_summary_.

    _extended_summary_

    Args:
    ----
        Encoder (_type_): _description_

    """

    def __init__(self, sigma: float, encoding_size: int) -> None:
        """_summary_.

        _extended_summary_

        Args:
        ----
            sigma (_type_): _description_
            input_size (_type_): _description_
            encoding_size (_type_): _description_

        """
        super().__init__(encoding_size=encoding_size)
        self.model = rff.layers.GaussianEncoding(
            sigma=sigma,
            input_size=1,
            encoded_size=encoding_size // 2,
        )

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_.

        _extended_summary_

        Args:
        ----
            src (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        encoding = self.model(src)

        return self.get_linspace_domain(torch.Size([src.size(0), 1, self.encoding_size])), encoding


class SineEncoder(Encoder):
    """_summary_.

    _extended_summary_

    Args:
    ----
        Encoder (_type_): _description_

    """

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """_summary_.

        _extended_summary_

        Args:
        ----
            src (_type_): _description_

        Returns:
        -------
            _type_: _description_

        """
        x = self.get_linspace_domain(torch.Size([src.size(0), 1, self.encoding_size]))

        return x, torch.sin(2 * torch.pi * src * x)


class EncodedFourierNeuralOperator(Operator):
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
        encoder: Encoder | None = None,
        encoding_size: int = 256,
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
            encoder (torch.nn.Module, optional): .
            encoding_size (int): _description_
            act (torch.nn.Module, optional): _description_. Defaults to None.
            device (torch.device, optional): _description_. Defaults to None.

        """
        super().__init__(shapes=shapes, device=device)

        self.encoding_size = encoding_size

        self.encoder: Encoder
        if encoder is None:
            self.encoder = RFFEncoder(sigma=1., encoding_size=encoding_size)
        else:
            self.encoder = encoder

        self.fno_shape = deepcopy(shapes)
        self.fno_shape.u.size = torch.Size((encoding_size,))
        self.fno_shape.u.dim = prod(shapes.u.size) * shapes.u.dim
        self.fno_shape.x.size = torch.Size((encoding_size,))
        self.fno_shape.x.dim = 1

        self.fno = FourierNeuralOperator(shapes=self.fno_shape, depth=depth, width=width, act=act, device=device)

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
        x, encoding = self.encoder(u.flatten(1, -1).unsqueeze(-1))

        return self.fno(x, encoding, y)
