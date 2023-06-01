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
from src.layers.conv import Conv
from src.cost import MSELoss

def test_convolution():
    # How to assign kernel for Conv layer
    # https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176/4

    # forward numpy
    batch_size = 2
    input_c, input_h, input_w = 3, 16, 24
    input_array = np.random.rand(batch_size, input_c, input_h, input_w)
    out_channel, in_channel, k_y, k_x = 8, 3, 3, 3
    kernel_numpy = np.random.rand(out_channel, in_channel, k_y, k_x)
    kernel_numpy = np.round(kernel_numpy, 4)
    conv_numpy = Conv(in_channel, out_channel, (k_y, k_x), kernel_weight=kernel_numpy)
    output_numpy = conv_numpy.forward(input_array)
    output_numpy = output_numpy.sum(axis=2)
    
    # forward torch
    input_torch = torch.from_numpy(input_array)
    kernel_torch = torch.from_numpy(kernel_numpy)
    conv = nn.Conv2d(out_channel, in_channel, (k_y, k_x), bias=False)
    input_torch.requires_grad = True
    with torch.no_grad():
        conv.weight = nn.Parameter(kernel_torch)
    output_torch = conv(input_torch)
    

    # backward numpy
    target_numpy = np.zeros(output_numpy.shape, dtype=np.float32)
    mse_loss_numpy = MSELoss()
    loss_numpy = mse_loss_numpy.forward(output_numpy, target_numpy)
    d_output = mse_loss_numpy.backward()
    d_weight, d_input = conv_numpy.backward(d_output)

    # backward torch
    target_torch = torch.from_numpy(target_numpy)
    target_torch = target_torch.double()
    l2_loss =  nn.MSELoss()
    loss_value_torch = l2_loss(output_torch, target_torch)
    loss_value_torch.backward()

    output_torch = output_torch.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch, output_numpy)
    print("Pass forward test")

    np.testing.assert_almost_equal(conv.weight.grad.detach().cpu().numpy(), d_weight)
    print("Pass backwared test for derivative of weight")

    np.testing.assert_almost_equal(input_torch.grad.detach().cpu().numpy(), d_input)
    print("Pass backwared test for derivative of input tensor")
    
def main():
    np.random.seed(42)
    test_convolution()

if __name__ == "__main__":
    main()