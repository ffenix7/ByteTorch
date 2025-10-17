
# --- BatchNorm & Dropout tests ---
import math
import numpy as np
import pytest
from src.core.tensor import Tensor
from src.nn.batchnorm import BatchNorm
from src.nn.dropout import Dropout

def test_batchnorm_forward_training():
    bn = BatchNorm(num_features=3)
    bn.training = True
    x = Tensor(np.random.randn(5, 3))
    out = bn(x)
    assert out.shape == x.shape
    # Sprawdź, czy średnia i wariancja są zbliżone do 0 i 1
    out_mean = out.data.mean(axis=0)
    out_var = out.data.var(axis=0)
    assert np.allclose(out_mean, 0, atol=1e-1)
    assert np.allclose(out_var, 1, atol=1e-1)

def test_batchnorm_forward_eval():
    bn = BatchNorm(num_features=3)
    bn.training = False
    x = Tensor(np.random.randn(5, 3))
    out = bn(x)
    assert out.shape == x.shape

def test_batchnorm_gradients():
    bn = BatchNorm(num_features=3)
    bn.training = True
    x = Tensor(np.random.randn(5, 3), requires_grad=True)
    out = bn(x)
    loss = out.sum()
    loss.backward()
    assert bn.gamma.grad is not None
    assert bn.beta.grad is not None

def test_dropout_training():
    dropout = Dropout(rate=0.5)
    dropout.training = True
    x = Tensor(np.ones((1000,)))
    out = dropout(x)
    # Oczekujemy, że ok. połowa elementów będzie wyzerowana
    zero_count = np.sum(out.data == 0)
    assert 400 < zero_count < 600  # tolerancja na losowość

def test_dropout_eval():
    dropout = Dropout(rate=0.5)
    dropout.training = False
    x = Tensor(np.ones((10,)))
    out = dropout(x)
    assert np.allclose(out.data, x.data)
