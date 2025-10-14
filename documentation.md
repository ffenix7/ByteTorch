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
  - [2. Neural Networks (nn) Module](#2-neural-networks-nn-module)
    - [Summary](#summary-1)
    - [Key Components](#key-components)
    - [Usage](#usage)
    - [Limitations](#limitations-1)
  - [3. Project Layout](#3-project-layout)
  - [4. API \& Utilities (Overview)](#4-api--utilities-overview)
  - [5. Testing and Examples](#5-testing-and-examples)
    - [Testing Guide: How to Write Tests for ByteTorch](#testing-guide-how-to-write-tests-for-bytetorch)
      - [Why Test?](#why-test)
      - [Tools and Setup](#tools-and-setup)
    - [How to Write Tests](#how-to-write-tests)
    - [Examples of Tests](#examples-of-tests)
      - [Running Tests](#running-tests)
  - [6. Development \& Contribution](#6-development--contribution)
    - [Setup](#setup)
    - [Contributing](#contributing)
  - [7. License \& Acknowledgements](#7-license--acknowledgements)
    - [License](#license)
    - [Acknowledgements](#acknowledgements)

---

## 1. Tensor

### Summary
- Lightweight NumPy-backed `Tensor` type with an autograd engine.
- File: `src/core/tensor.py`

Key properties
- `data` (np.array) — multidimensional array
- `dtype` - data type,
- `shape` - data shape,
- `ndim` - number of dimensions,
- `size` - number of values in data,
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
  - True Division (`/`): Supports Tensor / Tensor or Tensor / scalar
- **Matrix Operations**:
  -  Matrix multiplication (`@`): Tensor @ Tensor
- **Reduction** - returns a Tensor (scalar if `axis=None`, else reduced along axis): 
  - `mean(axis=None)`
  - `sum(axis=None)`
  - `min(axis=None)`
  - `max(axis=None)`
- **Shape Manipulation**:
  -  `transpose(axes=None)` - transpose matrix across  `axes`.
- **Indexing**:
  -  `t[idx]` returns a new Tensor with sliced data.
- **Assignment**:
  -  `t[idx] = value` modifies the underlying data.
- **Gradient Management**:
  -  `zero_grad()` resets gradients to zeros or `None`;
  -   `backward(grad=None)` performs backpropagation.

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
- No GPU support, no hooks, and limited broadcasting correctness in backward (uses simple `_unbroadcast`).
- `mean()` does not support `keepdims` yet.
- Only basic operations implemented; no advanced functions like `exp`, `log`, etc.
- The implementation is intentionally small and educational.

---

## 2. Neural Networks (nn) Module

### Summary
- Module for building neural network layers and models.
- File: `src/nn/module.py`
- Purpose: Neural network components with autograd integration.

### Key Components
- **`Module`**: Base class for all layers. Manages parameters (`parameters` list), `zero_grad()`, and provides `__call__` for easy use.
- **`Linear`**: Fully connected layer (`y = x @ W + b`). Inherits from `Module`.

### Usage
```python
from src.nn import Linear, ReLU

# Create layers
linear = Linear(in_features=10, out_features=5)
relu = ReLU()

# Forward pass
x = Tensor(np.random.randn(32, 10))
out = relu(linear(x))  # Shape: (32, 5)

# Training
loss = out.sum()
loss.backward()
linear.zero_grad()  # Reset gradients
```

### Limitations
- Minimal set of layers; no convolutions, pooling, etc.
- No optimizers yet (planned in `src/optim/`).
- Designed for simplicity and learning.

---

## 3. Project Layout

```
ByteTorch/
├── documentation.md          # This documentation
├── LICENSE                   # MIT License
├── README.md                 # Project overview and quick start
├── requirements.txt          # Python dependencies
│ 
├── activation/               # Activation functions
│   └── relu.py               # ReLU activation
│ 
├── src/                      # Source code
│   ├── core/
│   │   ├── __init__.py
│   │   └── tensor.py         # Tensor class implementation
│   ├── nn/                   # Neural network modules
│   │   ├── __init__.py       # Imports for nn components
│   │   ├── module.py         # Base Module class
│   │   ├── linear.py         # Linear layer
│   └── optim/                # Optimizers (planned, currently empty)
│ 
└── tests/
    ├── nn_test.py            # Tests for nn components
    └── tensor_test.py        # Unit tests for Tensor
```

---

## 4. API & Utilities (Overview)

Currently, the API is minimal and focused on the `Tensor` class. Future expansions will include:
- Optimizers in `src/optim/`.

No utilities are implemented yet.

---

## 5. Testing and Examples

### Testing Guide: How to Write Tests for ByteTorch

Testing is crucial for ensuring ByteTorch works correctly, especially with autograd and neural networks. This guide shows how to write tests using `pytest`.

#### Why Test?
- **Catch bugs**: Verify operations, gradients, and edge cases.
- **Regression prevention**: Ensure changes don't break existing code.
- **Documentation**: Tests serve as examples of usage.

#### Tools and Setup
- **Framework**: `pytest` (install via `pip install pytest`).
- **Files**: Tests in `tests/` (e.g., `tensor_test.py` for `Tensor`, `nn_test.py` for nn modules).
- **Structure**: Each test is a function starting with `test_`, using `assert` for checks.
- **Run tests**: `pytest` in project root, or `python -m pytest tests/`.

### How to Write Tests

1. **Import modules**:
   ```python
   import pytest
   import numpy as np
   #other libraries/modules needed to run test
   ```

2. **Best practices**:
   - **Descriptive names**: `test_tensor_mean_backward`.
   - **Isolate tests**: Each test independent.
   - **Use fixtures**: For repeated setup (e.g., random tensors).
   - **Approximate asserts**: `np.allclose` for floats, `pytest.approx`.
   - **Coverage**: Test all methods, error cases.
   - **Run often**: Test every change before commiting.

### Examples of Tests

**Tensor operations**:
```python
def test_tensor_mul():
    a = Tensor([2, 3])
    b = Tensor([4, 5])
    c = a * b
    assert np.array_equal(c.data, [8, 15])

def test_tensor_backward():
    x = Tensor(5.0, requires_grad=True)
    y = x ** 2
    y.backward()
    assert x.grad == 10.0  # dy/dx = 2x = 10
```

**NN modules**:
```python
def test_linear_backward():
    linear = Linear(3, 2)
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    out = linear(x)
    loss = out.sum()
    loss.backward()
    assert linear.weights.grad is not None
```

#### Running Tests
- `pytest tests/tensor_test.py` — specific file.
- `pytest -v` — verbose output.
- `pytest --cov=src` — coverage (install `pytest-cov`).

See full tests in `tests/tensor_test.py` and `tests/nn_test.py`. Add new tests for new features!

---

## 6. Development & Contribution

### Setup
1. Clone the repository: `git clone https://github.com/ffenix7/ByteTorch.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`

### Contributing
- Fork the repo and submit pull requests.
- Keep the code clean.
- Add tests for new features.

---

## 7. License & Acknowledgements

### License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

### Acknowledgements
- Inspired by PyTorch.
- Thanks to the open-source community for NumPy and Python.
- Thanks to everyone contributing in this project.