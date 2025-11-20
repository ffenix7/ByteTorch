from src.core.tensor import Tensor
class Sequential:
    def __init__(self, *layers):
        self.layers = layers
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    def train(self):
        for layer in self.layers:
            layer.training = True
    def eval(self):
        for layer in self.layers:
            layer.training = False
    def zero_grad(self):
        for param in self.parameters():
            param.grad = None