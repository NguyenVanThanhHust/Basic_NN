import unittest

import numpy as np
from src.layers.linear import Linear

class TestLayer(unittest.TestCase):
    def test_linear_forward(self):
        np.random.seed(42)
        batch_size = 2
        input_dim = 8
        output_dim = 4
        input_tensor = np.random.rand(batch_size, input_dim)
        linear_layer = Linear(input_dim, output_dim)
        true_output = np.dot(input_tensor, linear_layer.get_params()[0]) + linear_layer.get_params()[1]
        compute_output = linear_layer.forward(input_tensor)
        np.testing.assert_array_almost_equal(compute_output, true_output)