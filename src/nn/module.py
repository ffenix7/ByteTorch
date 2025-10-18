class Module:
    def __init__(self):
        self.params = []  
        self.training = True  

    def zero_grad(self):
        for param in self.params:
            if param.requires_grad:
                param.zero_grad()

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented!")

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self):
        return self.params