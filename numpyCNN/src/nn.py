import numpy as np
from src.layers.linear import Linear
from src.layers.activation import ReLU, Sigmoid

layers = {
    "linear": Linear, 
    "relu": ReLU, 
    "sigmoid": Sigmoid
}
class NeuralNetwork():
    def __init__(self, nn_arch):
        self.layers = []
        for each_layer in nn_arch:
            if each_layer["type"] == "linear":
                layer = Linear(input_dim=each_layer["input_dim"], output_dim=each_layer["output_dim"])
            elif each_layer["type"] == "relu":
                layer = ReLU(input_dim=each_layer["input_dim"])
            elif each_layer["type"] == "sigmoid":
                layer = Sigmoid(input_dim=each_layer["input_dim"])
            self.layers.append(layer)        

    def forward(self, input_array):
        X = input_array
        for layer in self.layers:
            output = layer.forward(X)
            X = output
        return output

    def backward(self, loss_derivative):
        return 

    def update_params(self):
        return 

    def __repr__(self, ):
        for layer in self.layers:
            print(layer, layer.in_dim, )
        return 
