from ..core.tensor import Tensor
from .module import Module

class Dropout(Module):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = Tensor.rand(x.shape)
            for i in range(mask.data.size):
                if mask.data.flat[i] < self.rate:
                    mask.data.flat[i] = 0.0
                else:
                    mask.data.flat[i] = 1.0
            return x * mask / (1.0 - self.rate)
        else:
            return x