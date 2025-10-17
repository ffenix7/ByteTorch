from ..core.tensor import Tensor
from .module import Module

class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, gamma=None, beta=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        if gamma is None:
            self.gamma = Tensor.ones(num_features, requires_grad=True)
        else:
            self.gamma = gamma       

        if beta is None:
            self.beta = Tensor.zeros(num_features, requires_grad=True)
        else:
            self.beta = beta

        self.running_mean = Tensor.zeros(num_features)
        self.running_var = Tensor.ones(num_features)

    def forward(self, x: Tensor):
        if self.training:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            x_normalized = (x - batch_mean) / (batch_var + self.eps).sqrt()
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            x_normalized = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        
        out = Tensor(self.gamma, requires_grad=self.gamma.requires_grad) * x_normalized + Tensor(self.beta, requires_grad=self.beta.requires_grad)
        return out