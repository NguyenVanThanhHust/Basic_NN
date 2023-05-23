import os, sys
import torch
import torch.nn as nn
import numpy as np
import math 

import matplotlib
import matplotlib.pyplot as plt

from loguru import logger

sys.path.append("..")
sys.path.append("../..")
from src.layers.linear import Linear
from src.cost import CrossEntropyLoss

def test_linear_forward():
    batch_size = 2
    input_dim = 8
    output_dim = 4
    input_array = np.random.rand(batch_size, input_dim)
    linear_numpy = Linear("linear", input_dim, output_dim)
    input_array = input_array.astype(np.float32)
    output_numpy = linear_numpy.forward(input_array)

    input_torch = torch.from_numpy(input_array)
    linear = nn.Linear(input_dim, output_dim, True)
    with torch.no_grad():
        linear.weight = nn.Parameter(torch.from_numpy(linear_numpy.w.T))
        linear.bias = nn.Parameter(torch.from_numpy(linear_numpy.b))
    output_torch = linear(input_torch)

    print(output_numpy)
    print(output_torch)
    output_torch = output_torch.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch, output_numpy)

def main():
    test_linear_forward()
    
if __name__ == "__main__":
    main()
    