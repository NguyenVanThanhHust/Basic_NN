class Layer:
    def __init__(self, name, input_dim, output_dim) -> None:
        """
        Initializes the layer
        
        Parameters:
        ----------
        input_dim: int or tuple
            Shape of the input data
        """
        raise NotImplementedError
    
    def forward(self, input, training):
        """
        Propagates forward
        
        Parameters:
        ----------
        input: numpy.array
        training: bool

        Return:
        numpy.array: output of this layer
        """
        raise NotImplementedError
    
    def backward(self, da):
        raise NotImplementedError

    def update_params(self, dw, db):
        raise NotImplementedError
    
    def get_params(self,):
        raise NotImplementedError

    def get_output_dim(self, ):
        raise NotImplementedError
    
    def __repr__(self) -> str:
        raise NotImplementedError