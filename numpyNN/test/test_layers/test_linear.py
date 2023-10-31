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
from src.cost import MSELoss

def test_linear():
    batch_size = 2
    input_dim = 8
    output_dim = 4

    # Forward numpy
    input_array = np.random.rand(batch_size, input_dim)
    linear_numpy = Linear("linear", input_dim, output_dim)
    input_array = input_array.astype(np.float32)
    output_numpy = linear_numpy.forward(input_array)
    target_numpy = np.zeros(output_numpy.shape, dtype=np.float32)
    
    # Forward torch
    input_torch = torch.from_numpy(input_array)
    linear = nn.Linear(input_dim, output_dim, True)
    with torch.no_grad():
        linear.weight = nn.Parameter(torch.from_numpy(linear_numpy.w.T))
        linear.bias = nn.Parameter(torch.from_numpy(linear_numpy.b))
    input_torch = input_torch.double()
    input_torch.requires_grad = True
    output_torch = linear(input_torch)

    # backward numpy
    mse_loss_numpy = MSELoss()
    loss_numpy = mse_loss_numpy.forward(output_numpy, target_numpy)
    d_output = mse_loss_numpy.backward()
    dw, db, d_input = linear_numpy.backward(d_output)
    
    # backward torch
    target_torch = torch.from_numpy(target_numpy)
    target_torch = target_torch.double()
    l2_loss =  nn.MSELoss()
    loss_value_torch = l2_loss(output_torch, target_torch)
    loss_value_torch.backward()

    output_torch = output_torch.detach().cpu().numpy()

    np.testing.assert_almost_equal(output_torch, output_numpy)
    print("Pass forward test")
    np.testing.assert_almost_equal(loss_value_torch.detach().cpu().numpy(), loss_numpy)
    print("Pass loss test")

    np.testing.assert_almost_equal(linear.weight.grad.detach().cpu().numpy(), dw.T)
    np.testing.assert_almost_equal(linear.bias.grad.detach().cpu().numpy(), db)
    print("Pass backward test for weight")
    
    np.testing.assert_almost_equal(input_torch.grad.detach().cpu().numpy(), d_input)
    print("Pass backwared test for derivative of input tensor")

def main():
    test_linear()
    
if __name__ == "__main__":
    main()
    