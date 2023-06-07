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
from src.cost import MSELoss, FakeL1Loss

def test_convolution_v2():
    # How to assign kernel for Conv layer
    # https://discuss.pytorch.org/t/setting-custom-kernel-for-cnn-in-pytorch/27176/4

    # forward numpy
    batch_size = 2
    input_c, input_h, input_w = 3, 16, 24
    input_array = np.random.rand(batch_size, input_c, input_h, input_w)
    out_channel, in_channel, k_y, k_x = 8, 3, 4, 3
    kernel_numpy = np.random.rand(out_channel, in_channel, k_y, k_x)
    kernel_numpy = np.round(kernel_numpy, 4)
    conv_numpy = Conv(3, 8, (4, 3), kernel_weight=kernel_numpy, reduction_method="mean")
    output_numpy = conv_numpy.forward(input_array)
    output_numpy = output_numpy.sum(axis=2)
    
    # forward torch
    input_torch = torch.from_numpy(input_array)
    kernel_torch = torch.from_numpy(kernel_numpy)
    conv = nn.Conv2d(8, 3, (4, 3), bias=False)
    input_torch.requires_grad = True
    with torch.no_grad():
        conv.weight = nn.Parameter(kernel_torch)
    output_torch = conv(input_torch)
    output_torch.retain_grad()

    # backward numpy
    target_numpy = np.zeros(output_numpy.shape, dtype=np.float32)
    mse_loss_numpy = MSELoss()
    loss_numpy = mse_loss_numpy.forward(output_numpy, target_numpy)
    d_output = mse_loss_numpy.backward()
    d_output = d_output / max(input_array.shape)
    d_weight, d_input = conv_numpy.backward(d_output)

    # backward torch
    target_torch = torch.from_numpy(target_numpy)
    target_torch = target_torch.double()
    l2_loss =  nn.MSELoss()
    loss_value_torch = l2_loss(output_torch, target_torch)
    loss_value_torch.backward()


    output_torch_np = output_torch.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch_np, output_numpy)
    print("Pass forward test")

    np.testing.assert_almost_equal(output_torch.grad.detach().cpu().numpy()*1144, d_output, decimal=5)
    print("Pass backwared test for derivative of output")

    np.testing.assert_almost_equal(conv.weight.grad.detach().cpu().numpy(), d_weight, decimal=5)
    print("Pass backwared test for derivative of weight")

    np.testing.assert_almost_equal(input_torch.grad.detach().cpu().numpy(), d_input, decimal=5)
    print("Pass backwared test for derivative of input tensor")



def test_convolution():
    # forward numpy
    batch_size = 1
    in_channel, input_h, input_w = 1, 6, 5
    input_array = np.random.rand(batch_size, in_channel, input_h, input_w)
    out_channel, k_y, k_x = 1, 3, 3
    kernel_numpy = np.random.rand(out_channel, in_channel, k_y, k_x)
    conv_numpy = Conv(in_channel, out_channel, (k_y, k_x), kernel_weight=kernel_numpy, reduction_method="sum")
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
    output_torch.retain_grad()

    # backward numpy
    target_numpy = np.zeros(output_numpy.shape, dtype=np.float32)
    mse_loss_numpy = MSELoss()
    loss_numpy = mse_loss_numpy.forward(output_numpy, target_numpy)
    d_output = mse_loss_numpy.backward()
    d_output = d_output / max(input_array.shape)
    d_weight, d_input =  conv_numpy.backward(d_output)

    # backward torch
    target_torch = torch.from_numpy(target_numpy)
    target_torch = target_torch.double()
    l2_loss =  nn.MSELoss()
    loss_value_torch = l2_loss(output_torch, target_torch)
    loss_value_torch.retain_grad()
    loss_value_torch.backward()

    output_torch_np = output_torch.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch_np, output_numpy)
    print("Pass forward test")

    np.testing.assert_almost_equal(output_torch.grad.detach().cpu().numpy(), d_output, decimal=5)
    print("Pass backwared test for derivative of output")

    np.testing.assert_almost_equal(conv.weight.grad.detach().cpu().numpy(), d_weight, decimal=5)
    print("Pass backwared test for derivative of weight")

    np.testing.assert_almost_equal(input_torch.grad.detach().cpu().numpy(), d_input, decimal=5)
    print("Pass backwared test for derivative of input tensor")
    

def test_mse_loss():
    k = np.array(
        [
            [1,0,-1],
            [2,0,-2],
            [1,0,-1]
        ]).reshape(1,1,3,3).astype(np.float32)

    x = np.array(
    [
        [1,1,1,2,3],
        [1,1,1,2,3],
        [1,1,1,2,3],
        [2,2,2,2,3],
        [3,3,3,3,3],
        [4,4,4,4,4]
    ]).reshape(1,1,6,5).astype(np.float32)

    conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False,
        stride = 1,
        padding_mode='zeros',
        padding=0
    )


    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad = True
    conv.weight = torch.nn.Parameter(torch.from_numpy(k))
    output_torch = conv(x_tensor)
    output_torch.retain_grad()
    l2_loss =  nn.MSELoss()
    target_torch = torch.zeros(output_torch.shape)
    loss = l2_loss(output_torch, target_torch)

    loss.backward()

    in_channel, out_channel, input_h, input_w = 1, 1, 6, 5
    out_channel, k_y, k_x = 1, 3, 3
    conv_numpy = Conv(in_channel, out_channel, (k_y, k_x), kernel_weight=k, reduction_method="sum")
    output_numpy = conv_numpy.forward(x)
    output_numpy = output_numpy.sum(axis=2)
    
    # backward numpy
    target_numpy = np.zeros(output_numpy.shape, dtype=np.float32)
    mse_loss_numpy = MSELoss(reduction_method="mean")
    loss_numpy = mse_loss_numpy.forward(target_numpy, output_numpy)
    d_output = -mse_loss_numpy.backward() / max(x.shape)
    d_weight, d_input = conv_numpy.backward(d_output)
    
    # backward torch
    output_torch_np = output_torch.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch_np, output_numpy)
    print("Pass forward test")

    np.testing.assert_almost_equal(output_torch.grad.detach().cpu().numpy(), d_output, decimal=5)
    print("Pass backwared test for derivative of output")

    np.testing.assert_almost_equal(conv.weight.grad.detach().cpu().numpy(), d_weight, decimal=5)
    print("Pass backwared test for derivative of weight")

    np.testing.assert_almost_equal(x_tensor.grad.detach().cpu().numpy(), d_input, decimal=5)
    print("Pass backwared test for derivative of input tensor")
    
def test_convolution_manual():
    k = np.array(
        [
            [1,0,-1],
            [2,0,-2],
            [1,0,-1]
        ]).reshape(1,1,3,3).astype(np.float32)

    x = np.array(
    [
        [1,1,1,2,3],
        [1,1,1,2,3],
        [1,1,1,2,3],
        [2,2,2,2,3],
        [3,3,3,3,3],
        [4,4,4,4,4]
    ]).reshape(1,1,6,5).astype(np.float32)

    conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        bias=False,
        stride = 1,
        padding_mode='zeros',
        padding=0
    )

    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad = True
    conv.weight = torch.nn.Parameter(torch.from_numpy(k))
    out = conv(x_tensor)
    loss = out.sum()
    loss.backward()

    in_channel, out_channel, input_h, input_w = 1, 1, 6, 5
    out_channel, k_y, k_x = 1, 3, 3
    conv_numpy = Conv(in_channel, out_channel, (k_y, k_x), kernel_weight=k, reduction_method="sum")
    output_numpy = conv_numpy.forward(x)
    output_numpy = output_numpy.sum(axis=2)
    
    # backward numpy
    target_numpy = np.zeros(output_numpy.shape, dtype=np.float32)
    mse_loss_numpy = FakeL1Loss(reduction_method="sum")
    loss_numpy = mse_loss_numpy.forward(target_numpy, output_numpy)
    d_output = mse_loss_numpy.backward()
    d_weight, d_input = conv_numpy.backward(d_output)
    
    output_torch = out.detach().cpu().numpy()
    np.testing.assert_almost_equal(output_torch, output_numpy)
    print("Pass forward test")

    np.testing.assert_almost_equal(conv.weight.grad.detach().cpu().numpy(), d_weight)
    print("Pass backwared test for derivative of weight")

    np.testing.assert_almost_equal(x_tensor.grad.detach().cpu().numpy(), d_input)
    print("Pass backwared test for derivative of input tensor")

def main():
    np.random.seed(42)
    # test_convolution_manual()
    # test_mse_loss()
    # test_convolution()
    test_convolution_v2()

if __name__ == "__main__":
    main()