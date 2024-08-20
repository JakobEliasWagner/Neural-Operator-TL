import torch
from notl import RFFFourierNeuralOperator


class TestRFFFourierNeuralOperator:
    def test_can_initialize(self, sin_derivative_dataset):
        operator = RFFFourierNeuralOperator(sin_derivative_dataset.shapes)
        assert isinstance(operator, RFFFourierNeuralOperator)

    def test_can_forward(self, sin_derivative_dataset):
        operator = RFFFourierNeuralOperator(sin_derivative_dataset.shapes)

        for x, u, y, v in iter(sin_derivative_dataset):
            _ = operator(x, u, y)

        assert True
