import numpy as np
from src.layers.layer import Layer

class ReLU(Layer):
    def __init__(self, input_shape) -> None:
        """
        Initializes the layer
        
        Parameters:
        ----------
        input_shape: int or tuple
            Shape of the input data
        """
        self.input_shape = input_shape
        self.cache = None
    
    def forward(self, input_array, training):
        """
        Propagates forward
        
        Parameters:
        ----------
        input: numpy.array
        training: bool

        Return:
        numpy.array: output of this layer
        """
        out = np.copy(input_array)
        out[out < 0] = 0
        return out

    def backward(self, prev_d=None):
        assert self.cache, "Relu must have cache"
        out = np.ones(input_shape)
        out[self.cache < 0] = 0
        return out

    def update_params(self, dw, db):
        return 
    
    def get_params(self,):
        return

    def get_output_dim(self, ):
        return self.input_shape

class Sigmoid(Layer):
    def __init__(self, input_shape) -> None:
        """
        Initializes the layer
        
        Parameters:
        ----------
        input_shape: int or tuple
            Shape of the input data
        """
        self.input_shape = input_shape
        self.cache = None
    
    def forward(self, input_array, training):
        """
        Propagates forward
        
        Parameters:
        ----------
        input: numpy.array
        training: bool

        Return:
        numpy.array: output of this layer
        """
        out = 1 / (1 + np.exp(-input_array))
        self.cache = out
        return out

    def backward(self, ):
        out = np.ones(input_shape)
        out[self.cache < 0] = 0
        return out

    def update_params(self, dw, db):
        return 
    
    def get_params(self,):
        return

    def get_output_dim(self, ):
        return self.input_shape