class Module:
    def __init__(self):
        self.parameters = []  
        self.training = True  

    def zero_grad(self):
        for param in self.parameters:
            if param.requiers_grad:
                param.zero_grad()

    def forward(self, input):
        raise NotImplementedError("Forward method not implemented!")

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)