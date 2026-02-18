import numpy as np
from src.core.tensor import Tensor, Parameter
from src.nn.module import Module

class LayerNorm1D(Module):
    def __init__(self, size, eps = 1e-05):
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