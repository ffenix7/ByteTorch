from ...core.tensor import Tensor
from ..optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [0 for _ in params]
        
    def step(self):
        for i,param in enumerate(self.params):
            if param.requires_grad:
                param.ensure_grad()
                grad = param.grad
                self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grad
                param.data += self.velocity[i]