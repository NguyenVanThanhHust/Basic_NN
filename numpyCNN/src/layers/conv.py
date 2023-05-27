import numpy as np
from src.layers.layer import Layer

class Conv(Layer):
    def __init__(self, in_channel, \
            out_channel, \
            kernel_size, \
            stride=(1, 1), \
            padding=(0, 0, 0, 0), \
            name='conv', \
            kernel_weight=None, \
            ) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size) 
        self.stride = stride if type(stride) is tuple else (stride, stride)
        self.padding = padding
        self.kernel = None
        self.bias = None
        self.cache = {}
        self.kernel = kernel_weight
        self.name = name
        self.init()

    def init(self):
        if self.kernel is None:
            print("Initialize randomly")
            self.kernel = np.random.rand(self.out_channel, self.in_channel, self.kernel_size[0], self.kernel_size[1])
        else:
            print("Initialize from weight")
            assert self.kernel.shape[0] == self.out_channel
            assert self.kernel.shape[1] == self.in_channel
            assert self.kernel.shape[2] == self.kernel_size[0]
            assert self.kernel.shape[3] == self.kernel_size[1]
            
    def forward(self, input_tensor, training=True):
        batch_size = input_tensor.shape[0]
        input_tensor = self.pad(input_tensor)
        # Convolution

        batch_size, input_c, input_h, input_w = input_tensor.shape
        output_h = int((input_h - self.kernel_size[0])/self.stride[0]) + 1
        output_w = int((input_w - self.kernel_size[1])/self.stride[1]) + 1

        output_tensor = np.zeros((batch_size, self.out_channel, input_c, output_h, output_w))
        for image_index in range(batch_size):
            input_image = input_tensor[image_index, :, :, :]
            for channel in range(self.out_channel):
                current_kernel = self.kernel[channel, :, :, :]
                for h_index in range(output_h):
                    h_start = h_index * self.stride[0]
                    h_end = h_start + self.kernel_size[0]
                    for w_index in range(output_w):
                        w_start = w_index * self.stride[1]
                        w_end = w_start + self.kernel_size[1]
                        input_patch = input_tensor[image_index, :, h_start: h_end, w_start: w_end]
                        input_patch = input_patch[np.newaxis, np.newaxis, ...]
                        kernel = current_kernel[np.newaxis, np.newaxis, ...]
                        output_patch = input_patch * kernel
                        output_patch = output_patch.sum(axis=4)
                        output_patch = output_patch.sum(axis=3)
                        output_tensor[image_index, channel, :, h_index, w_index] = output_patch                             
        self.cache["input_tensor"] = input_tensor
        self.cache["output_tensor"] = output_tensor
        return output_tensor
    
    def backward(self, d_output):
        input_tensor = self.cache["input_tensor"]
        batch_size, in_channel, input_h, output_w = input_tensor.shape
        batch_size, out_channel, output_h, output_w = d_output.shape

        ## Calculate derivative of kernel
        d_weight = np.zeros((batch_size, out_channel, in_channel, self.kernel_size[0], self.kernel_size[1]), dtype=np.float32)
        for b in range(batch_size):
            for d in range(out_channel):
                for c in range(in_channel):
                    for ky_index in range(self.kernel_size[0]):
                        ky_start = ky_index* self.stride[0]
                        ky_end = ky_start + output_h
                        for kx_index in range(self.kernel_size[1]):
                            kx_start = kx_index* self.stride[1]
                            kx_end = kx_start + output_w
                            input_patch = input_tensor[b, c, ky_start: ky_end, kx_start: kx_end]
                            input_patch = input_patch[np.newaxis, np.newaxis, np.newaxis, :, :]                
                            kernel = d_output[b, d, :, :]
                            kernel = kernel[np.newaxis, np.newaxis, np.newaxis, :, :]
                            temp_d_weight = input_patch * kernel
                            d_weight[b, d, c, ky_index, kx_index] = temp_d_weight.mean()
        d_weight = d_weight.mean(0)
        print(self.kernel.shape,  d_weight.shape)            
          
        ## Calculate derivative of input
        d_input = np.zeros((batch_size, out_channel, in_channel, input_h, output_w), dtype=np.float32)
        
        return d_weight / batch_size, d_input
    
    def get_params(self):
        return self.kernel

    def update_params(self):
        return 
            
    def pad(self, x):
        top, bottom, left, right = self.padding
        x = np.pad(x, (top, bottom), mode='constant', constant_values=0)
        x = np.pad(x, (left, right), mode='constant', constant_values=0)
        return x
    
    def __repr__(self) -> str:
        conv_name = self.name
        conv_name = conv_name + " layer with kernel shape: ({}, {}, {})".format(self.out_channel, self.in_channel, self.kernel_size)
        return conv_name