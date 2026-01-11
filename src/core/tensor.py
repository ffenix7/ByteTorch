import numpy as np
from src.core.datatype import FLOAT32

class Tensor:
    def __init__(self, data, requires_grad=False, _prev=None, dtype=None):
        if dtype is not None:
            if not np.issubdtype(dtype, np.number):
                raise ValueError(f"Wrong dtype: {dtype}")
            self.data = self.data.astype(dtype)
            self.dtype = dtype
        else:
            if not np.issubdtype(self.data.dtype, np.number):
                raise ValueError(f"Data has non-numeric dtype: {self.data.dtype}")
            self.dtype = self.data.dtype
        
        if _prev is None:
            _prev = set()
        
        self.data = np.array(data, dtype=self.dtype)
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
        
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, dtype={self.dtype})"


    @staticmethod
    def randn(shape, requires_grad=False, dtype = FLOAT32):
        """Generates a tensor with random values from a normal distribution."""
        data = np.random.randn(*shape)
        return Tensor(data, requires_grad=requires_grad, dtype=dtype)
    
    @staticmethod
    def rand(shape, requires_grad=False, dtype = FLOAT32):
        """Generates a tensor with random values from a uniform distribution [0, 1)."""
        data = np.random.rand(*shape)
        return Tensor(data, requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def ones(shape, requires_grad=False, dtype = FLOAT32):
        data = np.ones(shape)
        return Tensor(data, requires_grad=requires_grad, dtype=dtype)

    @staticmethod
    def zeros(shape, requires_grad=False, dtype = FLOAT32):
        data = np.zeros(shape)
        return Tensor(data, requires_grad=requires_grad, dtype=dtype)

    #mathematical operations

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += -out.grad  # ∂(-x)/∂x = -1
            out._backward = _backward
            out._prev = {self}
        return out

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
    
    def __radd__(self, other):
        return self.__add__(other)
    
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

    def __rsub__(self, other):
        return -self.__sub__(other)
    
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

    def __rmul__(self, other):
        return self.__mul__(other)

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
    
    def __rtruediv__(self, other):
        if other == 0:
            raise ZeroDivisionError("Division by zero is not allowed!")
        out = Tensor(other / self.data, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += self._unbroadcast(out.grad * (-other / (self.data ** 2)), self.data.shape)  # ∂(c/x)/∂x = -c/(x^2)
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
    
    def __pow__(self, power):
        if isinstance(power, (int, float, np.integer, np.floating)):
            out = Tensor(self.data ** power, requires_grad=self.requires_grad)
            if self.requires_grad:
                def _backward():
                    self._ensure_grad()
                    self.grad += out.grad * (power * (self.data ** (power - 1)))  # ∂(x^p)/∂x = p*x^(p-1)
                out._backward = _backward
                out._prev = {self}
            return out
        else:
            raise NotImplementedError("Power only supports int or float as exponent.")
    
    def __rpow__(self, base):
        out = Tensor(base ** self.data, requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad * (np.log(base) * (base ** self.data))  # ∂(b^x)/∂x = ln(b)*b^x
            out._backward = _backward
            out._prev = {self}
        return out

    def sqrt(self):
        out = Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad * (0.5 / out.data)  # ∂(sqrt(x))/∂x = 1/(2*sqrt(x))
            out._backward = _backward
            out._prev = {self}
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad * out.data  # ∂(exp(x))/∂x = exp(x)
            out._backward = _backward
            out._prev = {self}
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad * (1 / self.data)  # ∂(log(x))/∂x = 1/x
            out._backward = _backward
            out._prev = {self}
        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad = self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad * (self.data / (out.data + 1e-8)) # ∂(abs(x))/∂x = sign(x)
            out._backward = _backward
            out._prev = {self}
        return out
    
    def sign(self):
        out = Tensor(np.sign(self.data), requires_grad = self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad * 0  # ∂(sign(x))/∂x = 0 
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
    
    def log_softmax(self, dim=-1):
        exps = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        softmax = exps / np.sum(exps, axis=dim, keepdims=True)
        out = Tensor(np.log(softmax), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad_output = out.grad
                sum_grad = np.sum(grad_output, axis=dim, keepdims=True)
                self.grad += grad_output - softmax * sum_grad  # ∂(log_softmax(x))/∂x
            out._backward = _backward
            out._prev = {self}
        return out
    
    #reduction

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad / np.prod(self.data.shape if axis is None else np.array(self.data.shape)[axis])
            out._backward = _backward
            out._prev = {self}
        return out

    def var(self, axis=None, keepdims=False): 
        out = Tensor(self.data.var(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                if axis is None:
                    n = self.data.size
                elif isinstance(axis, int):
                    n = self.data.shape[axis]
                else:
                    n = np.prod([self.data.shape[a] for a in axis])

                mean = self.data.mean(axis=axis, keepdims=True)
                self.grad += grad * 2 * (self.data - mean) / n # ∂(var(x))/∂x = 2*(x - mean)/n
            out._backward = _backward
            out._prev = {self}
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad  # ∂(sum(x))/∂x = 1
            out._backward = _backward
            out._prev = {self}
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad * (self.data == self.data.max(axis=axis, keepdims=True))
            out._backward = _backward
            out._prev = {self}
        return out

    def min(self, axis=None, keepdims=False):
        out = Tensor(self.data.min(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                grad = out.grad

                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)

                self.grad += grad * (self.data == self.data.min(axis=axis, keepdims=True))
            out._backward = _backward
            out._prev = {self}
        return out
    
    #Shape manipulation

    def reshape(self, size):
        out = Tensor(self.data, requires_grad=True, _prev={self})
        out.data = out.data.reshape(size)

        if self.requires_grad:
            def _backward():
                self._ensure_grad()
                self.grad += out.grad.reshape(self.shape)  # ∂(reshape(x))/∂x = reshape(1)
            out._backward = _backward 
        return out

    def flatten(self, start_dim = 0, end_dim = -1): #?: check
        out = Tensor(self.data, requires_grad=True, _prev={self}) 

        for i in range(start_dim, end_dim + 1):
            if i < 0:
                i += self.ndim
            
            new_shape = list(out.data.shape)
            new_shape[start_dim] = np.prod(new_shape[start_dim:end_dim + 1])
            
            del new_shape[start_dim + 1:end_dim + 1]
            out.data = out.data.reshape(new_shape)


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