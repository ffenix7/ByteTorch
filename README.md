
# ByteTorch

ByteTorch is a minimalistic deep learning framework, built from scratch in Python and inspired by PyTorch. It features a custom autograd engine, intuitive API, and a focus on clarity and educational value. The project demonstrates how core neural network concepts can be implemented in a concise and transparent way—without sacrificing flexibility.

## Key Features
- **Custom autograd engine** – Understand and extend automatic differentiation from the inside out.
- **Simple, readable codebase** – Designed for learning, tinkering, and rapid prototyping.
- **Core neural network layers** – Linear, activation functions, dropout, batch normalization, and more.
- **Optimizers** – SGD, Adam, and easy extensibility.
- **Jupyter notebook demos** – See ByteTorch in action on regression and classification tasks.

## Why ByteTorch?
ByteTorch is more than just a technical exercise—it's a showcase of practical engineering, curiosity, and a drive to deeply understand how modern ML frameworks work under the hood. The project is ideal for anyone who wants to:
- Explore the mechanics of deep learning from first principles
- See how autograd and neural network layers are built
- Use a lightweight, hackable alternative to big frameworks for small projects or teaching

## Example: Linear Regression in ByteTorch
```python
from src.core.tensor import Tensor
from src.nn.linear import Linear
from src.optim.optimizers.SGD import SGD

# Generate synthetic data
import numpy as np
X = np.random.randn(100, 1)
y = 3 * X + 2 + 0.1 * np.random.randn(100, 1)
X_tensor = Tensor(X, requires_grad=False)
y_tensor = Tensor(y, requires_grad=False)

# Define model and optimizer
model = Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(100):
	preds = model(X_tensor)
	loss = ((preds - y_tensor) ** 2).mean()
	model.zero_grad()
	loss.backward()
	optimizer.step()
```

## Notebooks
- [Linear Regression Example](notebooks/linear_regression.ipynb)
- [Classification Example](notebooks/classification_example.ipynb)

---
ByteTorch is a living project—open to ideas, improvements, and new challenges. Dive in, explore the code, and see what you can build!
