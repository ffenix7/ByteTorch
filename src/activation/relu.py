from src.core.tensor import Tensor
from src.nn.module import Module
import numpy as np

class ReLU(Module):
    """ReLU activation function."""
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of ReLU activation.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying ReLU.
        """
        out = Tensor(np.maximum(0, input.data), requires_grad=input.requires_grad)
        if input.requires_grad:
            def _backward():
                input.grad += (input.data > 0) * out.grad
            out._backward = _backward
            out._prev = {input}
        return out