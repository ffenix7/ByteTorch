from ..core.tensor import Tensor
from .module import Module

class LayerNorm1D(Module):
    def __init__(self, size, eps = 1e-05):
        super().__init__()
        self.size = size
        self.eps = eps

    def forward(self, input):
        mean = input.mean()
        std = input.std()
        
        output = Tensor((input - mean)/std + self.eps, requires_grad = input.requires_grad)
        return output