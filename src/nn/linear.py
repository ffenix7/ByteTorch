from ..core.tensor import Tensor
from .module import Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Tensor.randn((in_features, out_features), requires_grad=True)

        if bias:
            self.bias = Tensor.randn((out_features,), requires_grad=True)
        else:
            self.bias = None
        self.parameters = [self.weights]
        
        if self.bias is not None:
            self.parameters.append(self.bias)

    def forward(self, input):
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Input shape {input.shape} doesn't match in_features {self.in_features}")
        
        self.input = input
        output = input @ self.weights
        if self.bias is not None:
            output = output + self.bias
        return output