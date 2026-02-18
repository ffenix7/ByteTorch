from ..core.tensor import Tensor
from .module import Module

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embeddings = Tensor.randn((num_embeddings, embedding_dim), requires_grad = True)

    def forward(self, indices):
        return self.embeddings[indices]