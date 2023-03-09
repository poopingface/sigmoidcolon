import torch

from sigmoidcolon.functional import sigmoidcolon
from sigmoidcolon import SigmoidColon

def test_class():
    activation = SigmoidColon()
    x = torch.randn(10, 100, requires_grad=True)
    y = activation(x).sum()
    y.backward()

def test_functional():
    x = torch.randn(10, 100, requires_grad=True)
    y = sigmoidcolon(x).sum()
    y.backward()