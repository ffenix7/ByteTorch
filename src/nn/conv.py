from src.core.tensor import Tensor
from src.nn.module import Module

class Conv(Module):
    def __init__(self, in_features, out_features, stride = 1, padding = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stride = stride
        self.padding = padding
        self.weights = Tensor.randn((out_features, in_features), requires_grad=True)
        self.bias = Tensor.randn((out_features,), requires_grad=True)
        self.params = [self.weights, self.bias]

        
    def forward(self, input):
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Input shape {input.shape} doesn't match in_features {self.in_features}")
        self.input = input
        batch_size, seq_len, _ = input.shape
        out_seq_len = (seq_len + 2 * self.padding - 1) // self.stride + 1
        output = Tensor.zeros((batch_size, out_seq_len, self.out_features))

        if self.padding > 0:
            padded_input = Tensor.zeros((batch_size, seq_len + 2 * self.padding, self.in_features))
            padded_input[:, self.padding:self.padding + seq_len, :] = input
        else:
            padded_input = input

        for b in range(batch_size):
            for o in range(self.out_features):
                for i in range(out_seq_len):
                    start = i * self.stride
                    end = start + 1
                    output[b, i, o] = (padded_input[b, start:end, :] * self.weights[o, :]).sum() + self.bias[o]

        return output