import pytest
import torch
from continuiti.data import OperatorDataset


@pytest.fixture
def sin_derivative_dataset():
    x = torch.linspace(-1, 1, 23).reshape(1, 1, -1)
    u = torch.sin(x)

    y = torch.linspace(-1, 1, 29).reshape(1, 1, -1)
    v = torch.cos(y)

    return OperatorDataset(x=x, u=u, y=y, v=v)
