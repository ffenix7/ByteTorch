import numpy as np
import pytest
from src.core.tensor import Tensor
from src.nn.linear import Linear
from src.nn.activation.relu import ReLU

def test_linear_forward():
    """Test forward pass with bias."""
    linear = Linear(in_features=3, out_features=2)
    input = Tensor(np.random.randn(4, 3))
    output = linear(input)
    assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

def test_linear_forward_no_bias():
    """Test forward pass without bias."""
    linear = Linear(in_features=3, out_features=2, bias=False)
    input = Tensor(np.random.randn(4, 3))
    output = linear(input)
    assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"
    assert linear.bias is None

def test_linear_backward():
    """Test gradients with autograd."""
    linear = Linear(in_features=3, out_features=2)
    input = Tensor(np.random.randn(4, 3), requires_grad=True)
    output = linear(input)
    loss = output.sum()  
    loss.backward()
    
    assert linear.weights.grad is not None, "Weights grad should not be None"
    assert linear.bias.grad is not None, "Bias grad should not be None"
    assert linear.weights.grad.shape == linear.weights.shape
    assert linear.bias.grad.shape == linear.bias.shape

def test_linear_backward_no_bias():
    """Test gradients without bias."""
    linear = Linear(in_features=3, out_features=2, bias=False)
    input = Tensor(np.random.randn(4, 3), requires_grad=True)
    output = linear(input)
    loss = output.sum()
    loss.backward()
    
    assert linear.weights.grad is not None
    assert linear.bias is None

def test_linear_invalid_input_shape():
    """Test error for invalid input shape."""
    linear = Linear(in_features=3, out_features=2)
    input = Tensor(np.random.randn(4, 5))  # incorrect last dim
    with pytest.raises(ValueError, match="Input shape .* doesn't match in_features"):
        linear(input)

def test_linear_parameters():
    """Test parameters list."""
    linear = Linear(in_features=3, out_features=2)
    assert len(linear.parameters) == 2  # weights and bias
    assert linear.weights in linear.parameters
    assert linear.bias in linear.parameters
    
    linear_no_bias = Linear(in_features=3, out_features=2, bias=False)
    assert len(linear_no_bias.parameters) == 1  # tylko weights

def test_relu_forward():
    """Test ReLU forward pass."""
    relu = ReLU()
    input = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]))
    output = relu(input)
    expected = np.array([0.0, 0.0, 1.0, 2.0])
    assert np.allclose(output.data, expected)
    assert output.shape == input.shape

def test_relu_multidimensional():
    """Test ReLU on multidimensional tensor."""
    relu = ReLU()
    input = Tensor(np.array([[-1.0, 2.0], [0.0, -3.0]]))
    output = relu(input)
    expected = np.array([[0.0, 2.0], [0.0, 0.0]])
    assert np.allclose(output.data, expected)
    assert output.shape == input.shape

def test_relu_backward():
    """Test ReLU gradients with autograd."""
    relu = ReLU()
    input = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
    output = relu(input)
    loss = output.sum()
    loss.backward()
    
    # Gradient: 0 dla x<=0, 1 dla x>0
    expected_grad = np.array([0.0, 0.0, 1.0, 1.0])
    assert np.allclose(input.grad, expected_grad)

if __name__ == "__main__":
    pytest.main()
