import numpy as np

class Tensor:
    def __init__(self,data, requires_grad=False):
        self.data = np.array(data)
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size
        self.device = 'cpu'
        self.requires_grad = requires_grad
        self._prev = set()
        self._backward = lambda: None
        
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    #mathematical operations
    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

            def _backward():
                if self.requires_grad:
                    self.grad += out.grad # ∂(x+y)/∂x = 1
                if other.requires_grad:
                    other.grad += out.grad # ∂(x+y)/∂y = 1
            out._backward = _backward
            out._prev = {self, other}

        else:
            out = Tensor(self.data + other, requires_grad=self.requires_grad)

            def _backward():
                if self.requires_grad:
                    self.grad += out.grad # ∂(x+c)/∂x = 1

            out._backward = _backward
            out._prev = {self}
        return out
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
            
            def _backward():
                if self.requires_grad:
                    self.grad += out.grad  # ∂(x-y)/∂x = 1
                if other.requires_grad:
                    other.grad += out.grad * -1  # ∂(x-y)/∂y = -1

            out._backward = _backward
            out._prev = {self, other}
        else:
            out = Tensor(self.data - other, requires_grad=self.requires_grad)
            def _backward():
                if self.requires_grad:
                    self.grad += out.grad # ∂(x-y)/∂x = 1
 
            out._backward = _backward
            out._prev = {self}
        return out
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
            
            def _backward():
                if self.requires_grad:
                    self.grad = (self.grad or 0) + out.grad * other.data  # ∂(x*y)/∂x = y
                if other.requires_grad:
                    other.grad = (other.grad or 0) + out.grad * self.data  # ∂(x*y)/∂y = x
            
            out._backward = _backward
            out._prev = {self, other}
        else:
            out = Tensor(self.data * other, requires_grad=self.requires_grad)
            def _backward():
                if self.requires_grad:
                    self.grad += out.grad * other  # ∂(x*c)/∂x = c

            out._backward = _backward
            out._prev = {self}
        return out
        
    def __truediv__(self, other):
        if isinstance(other, Tensor):
            if np.any(other.data == 0):
                raise ZeroDivisionError("Division by zero is not allowed!")
            out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
            
            def _backward():
                if self.requires_grad:
                    self.grad += out.grad * (1 / other.data)  # ∂(x/y)/∂x = 1/y
                if other.requires_grad:
                    other.grad += out.grad * -self.data / (other.data ** 2)  # ∂(x/y)/∂y = -x/(y^2)

            out._backward = _backward
            out._prev = {self, other}
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed!")
            out = Tensor(self.data / other, requires_grad=self.requires_grad)

            def _backward():
                if self.requires_grad:
                    self.grad += out.grad * (1 / other) # ∂(x/c)/∂x = 1/c
            out._backward = _backward
            out._prev = {self}
        return out
        
    def mean(self):
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self.grad += out.grad * (1 / self.size)  # ∂(mean(x))/∂x = 1/N
            out._backward = _backward
            out._prev = {self}

        return out
        
    #indexing
    def __getitem__(self, idx):
        return Tensor(self.data[idx])
    
    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    #gradient 
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def backward(self, grad=None):
        if not self.requires_grad:
            raise ValueError("This tensor does not require gradients.")
        
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Gradients can only be implicitly created for scalar outputs.")
            grad = np.ones_like(self.data)
        
        self.grad += grad

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        # backward pass
        for t in reversed(topo):
            t._backward()
