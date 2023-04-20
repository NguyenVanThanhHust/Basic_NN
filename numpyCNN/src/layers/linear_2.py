# https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

import numpy as np

# Define network architecute
nn_arch = [
    {"in_dim": 2, "out_dim": 10, "activation": "relu"},
    {"in_dim", 10, "out_dim": 20, "activation", "relu"}, 
    {"in_dim", 20, "out_dim": 40, "activation", "relu"}, 
    {"in_dim", 40, "out_dim": 20, "activation", "relu"}, 
    {"in_dim", 40, "out_dim": 20, "activation", "relu"}, 
    {"in_dim", 20, "out_dim": 10, "activation", "relu"}, 
    {"in_dim", 10, "out_dim": 1, "activation", "sigmoid"}, 
]

def init_layers(nn_arch, random_seed=42):
    np.random.seed(random_seed)
    params_values = {}
    for idx, layer in enumerate(nn_arch):
        in_dim, out_dim = layer["in_dim"], layer["out_dim"]
        random_vector = np.random.rand(out_dim, in_dim)
        params_values["W_" + str(idx)] = random_vector*0.5
        params_values["b_" + str(idx)] = np.random.rand(output_dim, 1)
    return params_values 

def relu(Z):
    return np.maximum(0, Z)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1-sig)

def single_forward(input_tensor, weight, bias, activation_func="relu"):
    assert activation_func in ["relu", "sigmoid"], " not support activation function"
    if activation_func == "relu":
        act_func = relu
    elif activation_func == "sigmoid":
        act_func = sigmoid
    else:
        assert False
    output = np.dot(weight, input_tensor) + bias
    z_output = act_func(output)
    return z_output, output

def forward_full()