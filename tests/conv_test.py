# python
import numpy as np
import pytest
from src.nn.conv import Conv
from src.core.tensor import Tensor

def test_conv_forward_simple():
    # conv: in=2, out=1, stride=1, padding=0
    conv = Conv(in_features=2, out_features=1, stride=1, padding=0)
    # deterministic parameters
    conv.weights = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    conv.bias = Tensor(np.array([0.5]), requires_grad=True)
    conv.params = [conv.weights, conv.bias]

    # input: batch=1, seq_len=3, in_features=2
    x = Tensor(np.array([[[1.0, 0.0],
                          [0.0, 1.0],
                          [1.0, 1.0]]]))
    out = conv(x)
    assert out.shape == (1, 3, 1)
    expected = np.array([[[1.0 * 1.0 + 0.0 * 2.0 + 0.5],
                          [0.0 * 1.0 + 1.0 * 2.0 + 0.5],
                          [1.0 * 1.0 + 1.0 * 2.0 + 0.5]]])
    assert np.allclose(out.data, expected)

def test_conv_forward_padding():
    # conv: in=2, out=1, stride=1, padding=1
    conv = Conv(in_features=2, out_features=1, stride=1, padding=1)
    conv.weights = Tensor(np.array([[1.0, 1.0]]), requires_grad=True)
    conv.bias = Tensor(np.array([0.0]), requires_grad=True)
    conv.params = [conv.weights, conv.bias]

    # input: batch=1, seq_len=2
    x = Tensor(np.array([[[1.0, 2.0],
                          [3.0, 4.0]]]))
    out = conv(x)
    # padded sequence length = seq_len + 2*padding = 4 -> out_seq_len should be 4
    assert out.shape == (1, 4, 1)
    # expected: padding adds zeros at positions 0 and 3
    expected = np.array([[[0.0],            # dot([0,0]) + 0
                          [1.0 + 2.0],      # dot([1,2]) = 3
                          [3.0 + 4.0],      # dot([3,4]) = 7
                          [0.0]]])          # dot([0,0]) + 0
    assert np.allclose(out.data, expected)

def test_conv_forward_stride():
    # conv: in=2, out=1, stride=2, padding=0
    conv = Conv(in_features=2, out_features=1, stride=2, padding=0)
    conv.weights = Tensor(np.array([[1.0, 0.0]]), requires_grad=True)
    conv.bias = Tensor(np.array([0.0]), requires_grad=True)
    conv.params = [conv.weights, conv.bias]

    # input: batch=1, seq_len=5
    x = Tensor(np.array([[[1.0, 2.0],
                          [3.0, 4.0],
                          [5.0, 6.0],
                          [7.0, 8.0],
                          [9.0, 10.0]]]))
    out = conv(x)
    # out_seq_len = (5 - 1)//2 + 1 = 3
    assert out.shape == (1, 3, 1)
    expected = np.array([[[1.0],    # dot([1,2]) with [1,0] -> 1
                          [5.0],    # dot([5,6]) -> 5
                          [9.0]]])  # dot([9,10]) -> 9
    assert np.allclose(out.data, expected)

def test_conv_forward_input_shape_mismatch_raises():
    conv = Conv(in_features=3, out_features=1)
    # input last dim is 2 but conv expects 3
    x = Tensor(np.zeros((1, 4, 2)))
    with pytest.raises(ValueError):
        conv(x)