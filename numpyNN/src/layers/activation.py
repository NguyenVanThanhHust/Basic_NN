import os, sys
import numpy as np
from src.layers.layer import Layer

class ReLU(Layer):
    def __init__(self, input_dim, name="relu") -> None:
        """
        Initializes the layer
        
        Parameters:
        ----------
        input_dim: int or tuple
            Shape of the input data
        """
        self.input_dim = input_dim
        self.name = name
        self.cache = {}
    
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
        self.cache["input_tensor"] = input_array
        out[out < 0] = 0
        return out

    def backward(self, prev_d):
        out = np.ones(self.cache["input_tensor"].shape)
        out[self.cache["input_tensor"] < 0] = 0
        return out

    def update_params(self):
        return 
    
    def get_params(self,):
        return

    def get_output_dim(self, ):
        return self.input_dim

    def __repr__(self):
        return self.name  + " input dim: " + str(self.input_dim) + " output dim: " + str(self.input_dim)


class Sigmoid(Layer):
    def __init__(self, input_dim, name="sigmoid") -> None:
        """
        Initializes the layer
        
        Parameters:
        ----------
        input_dim: int or tuple
            Shape of the input data
        """
        self.input_dim = input_dim
        self.name = name
        self.cache = {}
    
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
        self.cache["input_tensor"] = out
        return out

    def backward(self, prev_d):
        out = self.cache["input_tensor"]*(1-self.cache["input_tensor"])
        return out

    def update_params(self,):
        return 
    
    def get_params(self,):
        return

    def get_output_dim(self, ):
        return self.input_dim

    def __repr__(self):
        return self.name  + " input dim: " + str(self.input_dim) + " output dim: " + str(self.input_dim)
