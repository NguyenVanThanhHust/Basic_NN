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

def test_convolution_forward():
    # How to assign kernel for Conv layer
    # https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176/4
    batch_size = 2
    input_c, input_h, input_w = 3, 16, 24
    input_array = np.random.rand(batch_size, input_c, input_h, input_w)
    
    input_torch = torch.from_numpy(input_array)
    
    out_channel, in_channel, k_y, k_x = 8, 3, 4, 3
    kernel_numpy = np.random.rand(out_channel, in_channel, k_y, k_x)
    kernel_numpy = np.round(kernel_numpy, 4)
    kernel_torch = torch.from_numpy(kernel_numpy)
    
    conv = nn.Conv2d(8, 3, (4, 3), bias=False)
    with torch.no_grad():
        conv.weight = nn.Parameter(kernel_torch)
        
    output_torch = conv(input_torch)
    numpy_conv = Conv(3, 8, (4, 3), kernel_weight=kernel_numpy)
    output_numpy = numpy_conv.forward(input_array)
    output_numpy = output_numpy.sum(axis=2)
    
    output_torch = output_torch.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch, output_numpy)

def main():
    np.random.seed(42)
    test_convolution_forward()
    return 

if __name__ == "__main__":
    main()