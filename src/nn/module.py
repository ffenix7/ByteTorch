import pickle

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
    
    def save(self, filename):
        params = [param.data for param in self.params]
        with open(filename, 'wb') as f:
            pickle.dump(params, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        for p, loaded in zip(self.parameters(), params):
            p.data = loaded