from abc import ABC, abstractmethod
from ..core.tensor import Tensor
class Loss(ABC):
    def __init__(self):
        self.input = None
        self.target = None
        self.loss = None

    @abstractmethod
    def forward(self, input, target):
        raise NotImplementedError("Forward method not implemented!")

    @abstractmethod
    def backward(self):
        raise NotImplementedError("Backward method not implemented!")
    
    def __call__(self, input, target):
        return self.forward(input, target)
    

class MSELoss(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        self.input = input
        self.target = target
        self.loss = ((input - target) ** 2).mean()
        return self.loss
    
    def backward(self):
        batch_size = self.input.shape[0]
        grad_input = (2 * (self.input - self.target)) / batch_size
        return grad_input
    
class L1Loss(Loss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target, reduction = 'mean'):
        self.input = input 
        self.target = target
        self.reduction = reduction

        diff = (input - target).abs()
        if self.reduction is None:
            self.loss = diff
        elif self.reduction == 'mean':
            self.loss = diff.mean()
        elif self.reduction == 'sum':
            self.loss = diff.sum()
        else:
            raise(ValueError("Invalid reduction type! Valid types are 'None', 'mean', 'sum'."))
        return self.loss
    
    def backward(self):
        grad_input = None
        if self.reduction is None:
            grad_input = (self.input - self.target).sign()
        elif self.reduction == 'mean':
            batch_size = self.input.shape[0]
            grad_input = (self.input - self.target).sign() / batch_size
        elif self.reduction == 'sum':
            grad_input = (self.input - self.target).sign()
        else:
            raise(ValueError("Invalid reduction type! Valid types are 'None', 'mean', 'sum'."))
        return grad_input