import random

import numpy as np
import pytest
import torch

pytest_plugins = ["tests.fixtures"]


@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(0)
    torch.manual_seed(0)
    np.random.default_rng(0)
