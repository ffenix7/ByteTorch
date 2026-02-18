import numpy as np
from src.core.tensor import Parameter
from src.nn.module import Module

class LayerNorm1D(Module):
    def __init__(self, size, eps = 1e-08):
        super().__init__()
        self.size = size
        self.eps = eps
        self.gamma = Parameter(np.ones(size))
        self.beta = Parameter(np.zeros(size))

    def forward(self, input):
        mean = input.mean()
        std = input.std()
        normalized = (input - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        return output

class LayerNorm2D(Module):
    """
    Following PyTorch implementation, data is by default normalised along second axis.
    Input size is supposed to be (B, C, L) or (B, C, W, H)
    """
    def __init__(self, size, eps = 1e-08, axis = 1):
        super().__init__()
        self.size = size
        self.axis = axis
        self.eps = eps
        self.gamma = Parameter.ones(size, requires_grad=True)
        self.beta = Parameter.zeros(size, requires_grad=True)

    def forward(self, input):
        mean = input.mean(axis = self.axis, keepdims = True)
        std = input.std(axis= self.axis, keepdims = True)
        normalized = (input - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        return output