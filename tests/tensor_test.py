import numpy as np
import pytest
from src.core.tensor import Tensor

def test_add():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x + y
    assert z.data == 5.0

    z.backward()
    assert x.grad == 1.0
    assert y.grad == 1.0

def test_sub():
    x = Tensor(5.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x - y
    assert z.data == 2.0

    z.backward()
    assert x.grad == 1.0
    assert y.grad == -1.0

def test_mul():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x * y
    assert z.data == 6.0

    z.backward()
    assert x.grad == 3.0
    assert y.grad == 2.0

def test_truediv():
    x = Tensor(6.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x / y
    assert z.data == 2.0

    z.backward()
    assert x.grad == pytest.approx(1/3)
    assert y.grad == pytest.approx(-6/9)

def test_scalar_mul_div():
    x = Tensor(4.0, requires_grad=True)
    z = x * 2
    assert z.data == 8.0
    z.backward()
    assert x.grad == 2.0

    x.zero_grad()
    z = x / 2
    assert z.data == 2.0
    z.backward()
    assert x.grad == 0.5

def test_mean():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    m = x.mean()
    assert m.data == 2.0

    m.backward()
    assert np.allclose(x.grad, [1/3, 1/3, 1/3])

def test_indexing():
    x = Tensor([10, 20, 30])
    assert x[0].data == 10
    x[1] = 99
    assert x.data[1] == 99

def test_complex_graph():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = (x + y) * (x - y) / y
    z.backward()
    # Ręcznie policzone gradienty:
    # z = (x+y)*(x-y)/y = (x^2 - y^2)/y
    # dz/dx = 2x / y = 4/3
    # dz/dy = -2y / y - (x^2 - y^2)/y^2 = -6/3 - (4-9)/9 = -2 - (-5/9) = -2 + 0.5555 = -1.4444
    assert pytest.approx(x.grad) == 4/3
    assert pytest.approx(y.grad) == -13/9

if __name__ == "__main__":
    pytest.main()
