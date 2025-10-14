import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _prev=set()):
        self.data = np.array(data)
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.size = self.data.size
        self.device = 'cpu'
        self.requires_grad = requires_grad
        self._prev = _prev
        self._backward = lambda: None
        
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    def randn(shape, requires_grad=False):
        data = np.random.randn(*shape)
        return Tensor(data, requires_grad=requires_grad)

    #mathematical operations
    def __add__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
            out._ensure_grad()

            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad , self.data.shape) # ∂(x+y)/∂x = 1
                if other.requires_grad:
                    other._ensure_grad()
                    other.grad += self._unbroadcast(out.grad , other.data.shape) # ∂(x+y)/∂y = 1
            out._backward = _backward
            out._prev = {self, other}

        else:
            out = Tensor(self.data + other, requires_grad=self.requires_grad)
            out._ensure_grad()

            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad , self.data.shape) # ∂(x+c)/∂x = 1

            out._backward = _backward
            out._prev = {self}
        return out
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
            out._ensure_grad()

            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad , self.data.shape)  # ∂(x-y)/∂x = 1
                if other.requires_grad:
                    other._ensure_grad()
                    other.grad += self._unbroadcast(out.grad * -1 , other.data.shape)  # ∂(x-y)/∂y = -1

            out._backward = _backward
            out._prev = {self, other}
        else:
            out = Tensor(self.data - other, requires_grad=self.requires_grad)
            out._ensure_grad()

            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad , self.data.shape)  # ∂(x-y)/∂x = 1
 
            out._backward = _backward
            out._prev = {self}
        return out
        
    def __mul__(self, other):
        if isinstance(other, Tensor):
            out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
            
            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad = self.grad + self._unbroadcast(out.grad * other.data, self.data.shape)  # ∂(x*y)/∂x = y
                if other.requires_grad:
                    other._ensure_grad()
                    other.grad = other.grad + self._unbroadcast(out.grad * self.data, other.data.shape)  # ∂(x*y)/∂y = x

            out._backward = _backward
            out._prev = {self, other}
        else:
            out = Tensor(self.data * other, requires_grad=self.requires_grad)
            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad * other, self.data.shape)  # ∂(x*c)/∂x = c

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
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad * (1 / other.data), self.data.shape)  # ∂(x/y)/∂x = 1/y
                if other.requires_grad:
                    other._ensure_grad()
                    other.grad += self._unbroadcast(out.grad * (-self.data / (other.data ** 2)), other.data.shape)  # ∂(x/y)/∂y = -x/(y^2)

            out._backward = _backward
            out._prev = {self, other}
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero is not allowed!")
            out = Tensor(self.data / other, requires_grad=self.requires_grad)

            def _backward():
                if self.requires_grad:
                    self._ensure_grad()
                    self.grad += self._unbroadcast(out.grad * (1 / other), self.data.shape)  # ∂(x/c)/∂x = 1/c
            
            out._backward = _backward
            out._prev = {self}
        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise ValueError("Matrix multiplication requires another Tensor.")
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.dot(other.data.T)  # ∂(x·y)/∂x = y^T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data.T.dot(out.grad)  # ∂(x·y)/∂y = x^T

        out._backward = _backward
        out._prev = {self, other}
        return out
        
    def mean(self, axis=None):  # TODO: add keepdims
        out = Tensor(self.data.mean(axis=axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad / np.prod(self.data.shape if axis is None else np.array(self.data.shape)[axis])
            out._backward = _backward
            out._prev = {self}
        return out
    
    def transpose(self, axes=None):
        out = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                if axes is None:
                    self.grad += out.grad.transpose()
                else:
                    inv_axes = np.argsort(axes)
                    self.grad += out.grad.transpose(inv_axes)
            out._backward = _backward
            out._prev = {self}
        return out
    
    #reduction

    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad  # ∂(sum(x))/∂x = 1
            out._backward = _backward
            out._prev = {self}
        return out
    
    def max(self, axis=None):
        out = Tensor(self.data.max(axis=axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad * (self.data == self.data.max(axis=axis, keepdims=True))
            out._backward = _backward
            out._prev = {self}
        return out
    
    def min(self, axis=None):
        out = Tensor(self.data.min(axis=axis), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad * (self.data == self.data.min(axis=axis, keepdims=True))
            out._backward = _backward
            out._prev = {self}
        return out
    

    #indexing
    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad, _prev=self._prev)
    
    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    #grad
    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

    def detach(self):
        return Tensor(self.data, requires_grad=False, _prev=set())

    def backward(self, grad=None):
        if not self.requires_grad:
            raise ValueError("This tensor does not require gradients.")
        
        if grad is None:
            if self.data.size != 1:
                raise ValueError("Gradients can only be implicitly created for scalar outputs.")
            grad = np.ones_like(self.data)
        
        self._ensure_grad()
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

        for t in reversed(topo):
            t._backward()

    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        Reduces grad to match target_shape by summing over broadcasted axes.
        Works for scalars, tensors with dimensions of 1, and multi-dimensional broadcast.
        """
        grad_shape = grad.shape
        target_ndim = len(target_shape)
        grad_ndim = len(grad_shape)

        while grad_ndim > target_ndim:
            grad = grad.sum(axis=0)
            grad_ndim -= 1

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad
    
    def _ensure_grad(self):
        if self.grad is None and self.requires_grad:
            self.grad = np.zeros_like(self.data)