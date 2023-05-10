import numpy as np
from src.layers.layer import Layer

class Conv(Layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=0, padding_type='none') -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_type = padding_type
        self.kernel = None
        self.bias = None
        self.cache = {}
        self.init()

    def init(self):
        self.kernel = np.random.rand((self.out_channel, self.kernel_size[0], self.kernel_size[1], self.in_channel))
        self.bias = np.zeros((1, 1, 1, self.out_channel))

        output_h = int()

    def forward(self, input_tensor, training):
        batch_size = input.shape[0]
        input_tensor = self.padding(input_tensor, self.padding_type)
        # Convolution

        batch_size, input_h, input_w, input_c = input_tensor.shape
        if self.padding_type == 'none':
            output_h = int((input_h - self.kernel_size[0])/self.stride + 1)
            output_w = int((input_w - self.kernel_size[1])/self.stride + 1)
        else:
            output_h = int((input_h - self.kernel_size[0] + 2)/self.stride + 1)
            output_w = int((input_w - self.kernel_size[1] + 2)/self.stride + 1)

        output_tensor = np.zeros((batch_size, self.out_channel, output_h, output_w))
        for image_index in batch_size:
            input_image = input_tensor[image_index, :, :, :]
            for channel in self.out_channel:
                current_kernel = self.kernel[channel, :, :, :]
                for h_index in range(output_h):
                    v_start = h_index * self.stride
                    v_end = v_start + self.stride

        return super().forward(input, training)
    
    def backward(self, da):
        return super().backward(da)
    
    def get_params(self):
        return self.kernel, self.bias

    def update_params(self):
        return 
            
    def get_output_dim(self):
        return self.out_channel
    
    def padding(self, x, padding_type='none'):
        assert padding_type in ['none', 'zero']
        if padding_type == 'none':
            return x
        if padding_type == 'zero':
            batch_size, h, w, c = x.shape
            tmp_x = np.zeros(batch_size, h+2, w+2, c)
            tmp_x[:, 1:-1, 1:-1, :] = x
            return tmp_x