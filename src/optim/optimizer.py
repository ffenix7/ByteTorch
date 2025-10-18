from src.core.tensor import Tensor

class Optimizer():
    def __init__(self, params: list[Tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= param.grad * self.lr

    def zero_grad(self):
        for param in self.params:
            if param.requires_grad:
                param.zero_grad()

