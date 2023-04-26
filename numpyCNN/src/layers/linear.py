import numpy as np
from src.layers.layer import Layer

class Linear(Layer):
    def __init__(self, name="linear", input_dim=None, output_dim=None) -> None:
        self.name = name
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.w = None
        self.b = None
        self.cache = {}
        self.init()

    def init(self):
        self.w = np.random.rand(self.out_dim, self.in_dim) * np.sqrt(2 / self.in_dim)
        self.b = np.zeros((1, self.out_dim))

    def forward(self, input_tensor, training=True):
        output_tensor = np.dot(input_tensor, self.w.T) + self.b.T
        if training:
            self.cache.update({'input_tensor': input_tensor, 'output_tensor':output_tensor})
        return output_tensor

    def backward(self, d_output):
        input_tensor, output_tensor = (self.cache[key] for key in {'input_tensor', 'output_tensor'})
        batch_size = input_tensor.shape[0]
        dw = 1/batch_size * np.dot(d_output.T, input_tensor)
        db = 1/batch_size * d_output.sum(axis=0, keepdims=True)
        d_input = np.dot(d_output, self.w)
        return d_input, dw, db

    def update_params(self, dw, db, alpha=0.1):
        self.w = self.w - alpha*dw
        self.b = self.b - alpha*db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.out_dim
    
    def __repr__(self, info) -> str:
        
        return super().__repr__()