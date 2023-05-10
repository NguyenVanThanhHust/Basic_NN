import numpy as np
from src.layers.layer import Layer

class Softmax(Layer):
    def __init__(self, in_dim) -> None:
        super().__init__(in_dim)