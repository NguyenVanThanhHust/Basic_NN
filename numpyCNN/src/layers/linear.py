import numpy as np
from src.layers.layer import Layer

class Linear(Layer):
    def __init__(self, in_dim, out_dim) -> None:
        np.random.seed(42)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = None
        self.b = None
        self.cache = {}
        self.init()

    def init(self):
        self.w = np.random.rand(self.out_dim, self.in_dim) * np.sqrt(2 / self.in_dim)
        self.b = np.zeros((1, self.out_dim))

    def forward(self, input_tensor, training=True):
        output_tensor = np.dot(input_tensor, self.w.T) + self.b.T
        if training:
            self.cache.update({'input_tensor': input_tensor, 'output_tensor':output_tensor})
        return output_tensor

    def backward(self, d_output):
        input_tensor, output_tensor = (self.cache[key] for key in {'input_tensor', 'output_tensor'})
        batch_size = input_tensor.shape[0]
        dw = 1/batch_size * np.dot(d_output.T, input_tensor)
        db = 1/batch_size * d_output.sum(axis=0, keepdims=True)
        d_input = np.dot(d_output, self.w)
        return d_input, dw, db

    def update_params(self, dw, db, alpha=0.1):
        self.w = self.w - alpha*dw
        self.b = self.b - alpha*db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.out_dim
    

if __name__ == "__main__":
    center_1 = (1, 1, 1)
    center_2 = (-1, -1, -1)
    num_sample = 100
    samples = []
    labels = []

    for i in range(num_sample):
        vector = np.random.rand(20)
        vector = vector[vector >= 0.5]
        samples.append(vector)
        labels.append(1)

    for i in range(num_sample):
        vector = np.random.rand(20)
        vector = vector[vector < 0.5]
        samples.append(vector)
        labels.append(0)

    labels = np.array(labels)
    samples = np.array(samples)
    linear_layer = Linear(20, 2)
    acc = 0.0
    epoch = 50
    batch_size = 4
    for i in range(epoch):
        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)
        num_iter =2 * num_sample / batch_size
        for iter in num_iter:
            start_index = iter * batch_size
            end_index = (iter + 1) * batch_size
            input_vectors = samples[start_index: end_index]
            input_labels = samples[start_index: end_index]
            pred_labels = linear_layer.forward(input_vectors)            
            d_output = pred_labels - input_labels
            d_input, dw, db = linear_layer.backward(d_output)
            linear_layer.update_params(dw, db)
            
