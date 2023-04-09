import numpy as np
from src.layers.layer import Layer

class FullyConnected(Layer):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__(in_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = None
        self.b = None
        self.cache = {}

    def init(self):
        self.w = np.random.rand(self.in_dim, self.out_dim) * np.sqrt(2 / self.in_dim)
        self.b = np.zeros(1, self.out_dim)

