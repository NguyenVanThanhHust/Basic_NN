import numpy as np
from src.layers.layer import Layer

class Conv(Layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=0, padding=None) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        assert len(self.in_dim)==4, 'input dim must be in format (B, H, W, C)'
        assert len(self.kernel_size)==4, 'kernel size must have 4 channels'
        assert self.in_dim[-1] == self.kernel_size[-1], 'kernel and input must have same number of channel'
        self.w = None
        self.b = None
        self.cache = {}
        self.init()

    def init(self):
        self.w = np.random.rand(self.kernel_size) * np.sqrt(2 / self.in_dim)
        self.b = np.zeros((1, self.out_dim))

    def forward(self, input, training):
        batch_size = input.shape[0]

        return super().forward(input, training)
    
    def backward(self, da):
        return super().backward(da)
    
    def get_params(self):
        return super().get_params()
    
    def get_output_dim(self):
        return super().get_output_dim()
    