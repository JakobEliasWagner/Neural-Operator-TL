from math import prod

import pytest
from continuiti.operators.losses import MSELoss
from continuiti.trainer import Trainer
from notl import EncodedFourierNeuralOperator, RFFEncoder, SineEncoder


@pytest.fixture
def sine_encoder():
    return SineEncoder(
        encoding_size=32,
    )


@pytest.fixture
def rff_encoder():
    return RFFEncoder(
        sigma=1.0,
        encoding_size=32,
    )


@pytest.fixture
def none_encoder():
    return None


@pytest.mark.parametrize("encoder", ["sine_encoder", "rff_encoder", "none_encoder"])
class TestEncodedFourierNeuralOperator:
    def test_can_initialize(self, sin_derivative_dataset, encoder, request):
        encoder = request.getfixturevalue(encoder)
        operator = EncodedFourierNeuralOperator(sin_derivative_dataset.shapes, encoder=encoder)
        assert isinstance(operator, EncodedFourierNeuralOperator)

    def test_can_forward(self, sin_derivative_dataset, encoder, request):
        encoder = request.getfixturevalue(encoder)
        operator = EncodedFourierNeuralOperator(sin_derivative_dataset.shapes, encoder=encoder)

        for x, u, y, _ in iter(sin_derivative_dataset):
            x_b, u_b, y_b = x.unsqueeze(1), u.unsqueeze(1), y.unsqueeze(1)  # mock batch size 1
            _ = operator(x_b, u_b, y_b)

        assert True

    def test_can_overfit(self, sin_derivative_dataset, encoder, request):
        encoder = request.getfixturevalue(encoder)
        operator = EncodedFourierNeuralOperator(sin_derivative_dataset.shapes, encoder=encoder)

        Trainer(operator).fit(sin_derivative_dataset, tol=1e-3)

        x, u, y, v = (
            sin_derivative_dataset.x,
            sin_derivative_dataset.u,
            sin_derivative_dataset.y,
            sin_derivative_dataset.v,
        )
        loss = MSELoss()(operator, x, u, y, v)
        threshold = 5e-3
        assert loss < threshold
