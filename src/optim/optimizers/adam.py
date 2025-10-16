from ..optimizer import Optimizer

class Adam(Optimizer):
    def __init(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [0 for _ in params]
        self.v = [0 for _ in params]
        self.t = 0
        
    def step(self):
        self.t += 1
        for i,param in enumerate(self.params):
                if param.requires_grad:
                    grad = param.grad
                    self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
                    self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
                    
                    m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                    v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                    
                    param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)