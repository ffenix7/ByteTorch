# ByteTorch — Project Documentation

This document describes the ByteTorch project: a minimal, educational NumPy-backed tensor library with a small reverse-mode autograd engine. The goal is to provide a compact, easy-to-read implementation for learning and experimentation.

Table of contents
- Tensor (core)
- Project layout
- API & utilities (overview)
# ByteTorch — Project Documentation

This document describes the ByteTorch project: a minimal, educational NumPy-backed tensor library with a small reverse-mode autograd engine. The goal is to provide a compact, easy-to-read implementation for learning and experimentation.

Table of contents
- Tensor (core)
- Project layout
- API & utilities (overview)
- Testing and examples
- Development & contribution
- License & acknowledgements

---

## 1. Tensor (core)

Summary
- Lightweight NumPy-backed `Tensor` type with a small reverse-mode autograd engine.
- File: `src/core/tensor.py`
- Purpose: educational experiments and simple autograd examples.

Key properties
- `data` (np.ndarray) — underlying values
- `dtype`, `shape`, `ndim`, `size` — derived from `self.data`
- `device` — `'cpu'`
- `requires_grad` — whether to track gradients
- `grad` — NumPy array with accumulated gradients (when required)
- `_prev`, `_backward` — autograd internals used to build and traverse the compute graph

Construction
- Signature: `Tensor(data, requires_grad=False)`

Supported operations (implemented)
- Elementwise: add (`+`), sub (`-`), mul (`*`), true division (`/`) with scalars or `Tensor` operands
- Reduction: `mean()` -> scalar `Tensor`
- Indexing and assignment: `t[idx]`, `t[idx] = value`
- Backprop: `tensor.backward(grad=None)` and `tensor.zero_grad()`

Autograd notes
- Each op constructs an output `Tensor` and sets `out._prev` (parents) and `out._backward` (closure that accumulates gradients into parents).
- `backward()` creates a topological ordering of nodes reachable from the output and calls `_backward()` in reverse order.
- Implicit `backward()` without a `grad` argument is permitted only when the output is scalar (size == 1).
- Broadcasting-aware gradient reduction is minimal in the current implementation.

Examples

Scalar example:

```python
from src.core.tensor import Tensor

a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)
c = a * b + a
c.backward()
print(a.grad, b.grad)
```

Mean on vector:

```python
from src.core.tensor import Tensor
import numpy as np

x = Tensor(np.array([1.0, 3.0, 5.0]), requires_grad=True)
m = (x * 2.0).mean()
m.backward()
print(x.grad)
```

Limitations
- No GPU support, no hooks, and limited broadcasting correctness in backward. The implementation is intentionally small and educational.


