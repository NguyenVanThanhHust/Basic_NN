import os, sys

import numpy as np
from src.layers.linear import Linear
from src.layers.activation import ReLU, Sigmoid


class NeuralNetwork():
    def __init__(self, nn_arch):
        self.layers = []
        for idx, each_layer in enumerate(nn_arch):
            if each_layer["type"] == "linear":
                layer = Linear(input_dim=each_layer["input_dim"], \
                                output_dim=each_layer["output_dim"], 
                                name="linear_" + str(idx))
            elif each_layer["type"] == "relu":
                layer = ReLU(input_dim=each_layer["input_dim"], \
                                name="relu_" + str(idx))
            elif each_layer["type"] == "sigmoid":
                layer = Sigmoid(input_dim=each_layer["input_dim"], \
                                name="sigmoid_" + str(idx))
            self.layers.append(layer)        

    def forward(self, input_array):
        X = input_array
        for layer in self.layers:
            output = layer.forward(X, training=True)
            X = output
        return output

    def backward(self, loss_derivative):
        prev_derivative = loss_derivative
        for layer in reversed(self.layers):
            derivative = layer.backward(prev_derivative)
            prev_derivative = derivative
            

    def update_params(self):
        for layer in reversed(self.layers):
            layer.update_params()

    def __repr__(self, ):
        for layer in self.layers:
            print(layer)
        return "main network"
