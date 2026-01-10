from abc import ABC, abstractmethod
from ..core.tensor import Tensor
import numpy as np

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
    def __init__(self, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input, target):
        self.input = input 
        self.target = target

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
    
class CrossEntropyLoss(Loss):
    def __init__(self, weight = None, reduction = 'mean', label_smoothing = 0.0):
        super().__init__()
        if weight is None:
            self.weight = 1
        else:
            self.weight = weight
        if label_smoothing < 0.0 or label_smoothing > 1.0 or not isinstance(label_smoothing, float):
            self.label_smoothing = 0.0
            raise ValueError("Label smoothing must be in the range [0.0, 1.0]")
        else:
            self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input, target):
        if input.ndim != 2:
            raise ValueError("CrossEntropyLoss currently supports only 2D input (N, C).")
        
        log_probs = input.log_softmax(axis=1)
        n_classes = input.shape[1]
        smooth_target = (1 - self.label_smoothing) * target + self.label_smoothing / n_classes
        self.loss = - (smooth_target * log_probs).sum(axis=1)

        if self.reduction == 'mean':
            self.loss = self.loss.mean()
        elif self.reduction == 'sum':
            self.loss = self.loss.sum()
        
        return self.loss