import os, sys
import numpy as np

class Dropout(Layer):
    def __init__(self, prob=0.5, name="drop_out") -> None:
        """
        Initializes the layer
        
        Parameters:
        ----------
        input_dim: int or tuple
            Shape of the input data
        """
        self.prob = prob
        self.name = name
        self.cache = None 
        
    def forward(self, input, training=True):
        """
        Propagates forward
        
        Parameters:
        ----------
        input: numpy.array
        training: bool

        Return:
        numpy.array: output of this layer
        """
        if training:
            input_shape = input.shape
            random_matrix = np.random.rand(input_shape)
            random_matrix = random_matrix.round()
            self.cache = random_matrix
            return np.multiply(input, random_matrix)
        else:
            return input 
               
    def backward(self, da):
        return np.multiply(self.cache, da) 

    def update_params(self):
        raise NotImplementedError
    
    def get_params(self,):
        raise NotImplementedError

    def get_output_dim(self, ):
        return self.cache.shape
    
    def __repr__(self) -> str:
        raise NotImplementedError