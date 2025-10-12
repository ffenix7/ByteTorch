# ByteTorch — Project Documentation

Let me introduce to you ByteTorch - fully working? (maybe one day), optimised? (you wish) and lightweight (this one is true) wersion of PyTorch. <br><br>
**Brought to you by  [ffenix7](github.com/ffenix7)**

## Table of Contents
- [ByteTorch — Project Documentation](#bytetorch--project-documentation)
  - [Table of Contents](#table-of-contents)
  - [1. Tensor](#1-tensor)
    - [Summary](#summary)
    - [Supported Operations](#supported-operations)
    - [Autograd Notes](#autograd-notes)
    - [Internal Methods](#internal-methods)
    - [Examples](#examples)
    - [Limitations](#limitations)
  - [2. Project Layout](#2-project-layout)
  - [3. API \& Utilities (Overview)](#3-api--utilities-overview)
  - [4. Testing and Examples](#4-testing-and-examples)
    - [Testing](#testing)
    - [Examples](#examples-1)
  - [5. Development \& Contribution](#5-development--contribution)
    - [Setup](#setup)
    - [Contributing](#contributing)
  - [6. License \& Acknowledgements](#6-license--acknowledgements)
    - [License](#license)
    - [Acknowledgements](#acknowledgements)

---

## 1. Tensor

### Summary
- Lightweight NumPy-backed `Tensor` type with a small reverse-mode autograd engine.
- File: `src/core/tensor.py`

Key properties
- `data` (np.array) — multidimensional array
- `dtype`, `shape`, `ndim`, `size` — derived from `self.data`
- `device` — `'cpu'` (hardcoded, no GPU support yet)
- `requires_grad` — boolean flag to track gradients
- `grad` — NumPy array with accumulated gradients (initialized to zeros if requires_grad=True, else None)
- `_prev` — set of parent tensors in the computation graph
- `_backward` — function to compute and accumulate gradients during backpropagation

Construction
- Signature: `Tensor(data, requires_grad=False, _prev=set())`
- `data` can be any array-like object convertible to NumPy array.
- `_prev` is internal, used for graph construction.

### Supported Operations
- **Elementwise Arithmetic**:
  - Addition (`+`): Supports Tensor + Tensor or Tensor + scalar.
  - Subtraction (`-`): Supports Tensor - Tensor or Tensor - scalar.
  - Multiplication (`*`): Supports Tensor * Tensor or Tensor * scalar.
  - True Division (`/`): Supports Tensor / Tensor or Tensor / scalar (raises `ZeroDivisionError` on division by zero).
- **Reduction**: `mean(axis=None)` → Returns a Tensor (scalar if `axis=None`, else reduced along axis). *Note: `keepdims` not yet implemented (TODO).*
- **Indexing**: `t[idx]` returns a new Tensor with sliced data.
- **Assignment**: `t[idx] = value` modifies the underlying data.
- **Gradient Management**: `zero_grad()` resets gradients to zeros or `None`; `backward(grad=None)` performs backpropagation.

### Autograd Notes
- Each operation creates a new output Tensor with `_prev` set to input tensors and `_backward` set to a closure that computes gradients.
- `backward()` builds a topological sort of the computation graph and calls `_backward` in reverse order.
- If `grad` is `None`, it's assumed to be `ones_like(data)`, but only allowed for scalar tensors (`size == 1`).
- Broadcasting is handled via `_unbroadcast()` method, which reduces gradients to match tensor shapes by summing over broadcasted dimensions.
- Gradients are accumulated in-place using `+=`.

### Internal Methods
- `_unbroadcast(grad, target_shape)`: Static method to handle gradient broadcasting reduction.
- `_ensure_grad()`: Ensures `grad` is initialized if `requires_grad=True`.

### Examples

**Scalar Example:**
```python
from src.core.tensor import Tensor

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a * b + a
c.backward()
print(a.grad, b.grad)  # Output: 4.0 2.0
```

**Mean on Vector:**
```python
from src.core.tensor import Tensor
import numpy as np

x = Tensor(np.array([1.0, 3.0, 5.0]), requires_grad=True)
m = (x * 2.0).mean()
m.backward()
print(x.grad)  # Output: [0.66666667 0.66666667 0.66666667]
```

**Broadcasting Example:**
```python
from src.core.tensor import Tensor
import numpy as np

a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)  # shape (1, 2)
b = Tensor(np.array([[3.0], [4.0]]), requires_grad=True)  # shape (2, 1)
c = a + b  # Broadcasting to (2, 2)
c.backward(np.ones_like(c.data))
print(a.grad)  # Gradients summed over broadcasted dims
print(b.grad)
```

### Limitations
- No GPU support, no hooks
- `mean()` does not support `keepdims` yet.
- Only basic operations implemented; no advanced functions like `exp`, `log` yet

---

## 2. Project Layout

```
ByteTorch/
├── documentation.md          # This documentation
├── LICENSE                   # MIT License
├── README.md                 # Project overview and quick start
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── core/
│   │   ├── __init__.py
│   │   └── tensor.py         # Tensor class implementation
│   ├── nn/                   # Neural network modules (planned, currently empty)
│   └── optim/                # Optimizers (planned, currently empty)
└── tests/
    ├── tensor_test.py        # Unit tests for Tensor
    └── __pycache__/          # Python cache
```

---

## 3. API & Utilities (Overview)

Currently, the API is minimal and focused on the `Tensor` class. Future expansions will include:
- Neural network layers in `src/nn/`.
- Optimizers in `src/optim/`.

No utilities are implemented yet.

---

## 4. Testing and Examples

### Testing
- Tests are located in `tests/tensor_test.py`.
- Run tests with `pytest` or `python -m pytest`.
- Dependencies: Listed in `requirements.txt`.

### Examples
See the examples in the [Tensor section](#1-tensor-core) above. More examples may be added in the future.

---

## 5. Development & Contribution

### Setup
1. Clone the repository: `git clone https://github.com/ffenix7/ByteTorch.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`

### Contributing
- Fork the repo and submit pull requests.
- Keep the code clean.
- Add tests for new features.

---

## 6. License & Acknowledgements

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Acknowledgements
- Inspired by PyTorch.
- Thanks to the open-source community for NumPy and Python.