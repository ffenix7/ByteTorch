import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.core.tensor import Tensor
from src.nn.sequential import Sequential

class DummyLayer:
    def __init__(self):
        self.training = True
        self._param = Tensor(np.array([1.0]), requires_grad=True)
    def __call__(self, x):
        return x + 1
    def parameters(self):
        return [self._param]

def test_sequential_forward():
    seq = Sequential(DummyLayer(), DummyLayer())
    x = Tensor(np.array([0.0]))
    out = seq(x)
    assert np.allclose(out.data, np.array([2.0]))

def test_sequential_parameters():
    l1 = DummyLayer()
    l2 = DummyLayer()
    seq = Sequential(l1, l2)
    params = seq.parameters()
    assert l1._param in params and l2._param in params
    assert len(params) == 2

def test_sequential_train_eval():
    l1 = DummyLayer()
    l2 = DummyLayer()
    seq = Sequential(l1, l2)
    seq.eval()
    assert not l1.training and not l2.training
    seq.train()
    assert l1.training and l2.training

def test_sequential_zero_grad():
    l1 = DummyLayer()
    l2 = DummyLayer()
    seq = Sequential(l1, l2)
    for p in seq.parameters():
        p.grad = np.array([42.0])
    seq.zero_grad()
    for p in seq.parameters():
        assert p.grad is None