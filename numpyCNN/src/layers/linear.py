import os, sys
import numpy as np
from src.layers.layer import Layer

class Linear(Layer):
    def __init__(self, name="linear", input_dim=None, output_dim=None) -> None:
        self.name = name
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.w = None
        self.b = None
        self.name = name
        self.cache = {}
        self.init()

    def init(self):
        self.w = np.random.rand(self.in_dim, self.out_dim) * np.sqrt(2 / self.in_dim)
        self.b = np.zeros((1, self.out_dim))

    def forward(self, input_tensor, training=True):
        output_tensor = np.dot(input_tensor, self.w) + self.b
        if training:
            self.cache["input_tensor"] = input_tensor
            self.cache["output_tensor"] = output_tensor
        return output_tensor

    def backward(self, d_output):
        input_tensor = self.cache["input_tensor"]
        output_tensor = self.cache["output_tensor"]
        batch_size = input_tensor.shape[0]
        dw = 1/batch_size * np.dot(input_tensor.T, d_output)
        db = 1/batch_size * d_output.sum(axis=0, keepdims=True)
        assert dw.shape == self.w.shape, "expect same shape, get {} for self.w and {} for dw".format(self.w.shape, dw.shape)
        self.cache.update({'dw':dw, 'db':db})
        return dw

    def update_params(self, alpha=0.01):
        try:
            dw, db = self.cache['dw'], self.cache['db']
            self.w = self.w - alpha*dw
            self.b = self.b - alpha*db
        except Exception as e:
            print(self.w.shape, self.b.shape, dw.shape, db.shape)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit()

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.out_dim
    
    def __repr__(self) -> str:
        if "input_tensor" in self.cache.keys():
            return self.name + " input dim: " + str(self.in_dim) + " output dim: " + str(self.out_dim) + " input tensor shape: " + str(self.cache["input_tensor"].shape)
        else:
            return self.name + " input dim: " + str(self.in_dim) + " output dim: " + str(self.out_dim) 