# https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/03_numpy_neural_net
# Build dataset
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from src.nn import NeuralNetwork

TRAIN_SIZE, TEST_SIZE = 2000, 400
RANDOM_SEED = 42

X_data, y_label = make_moons((TRAIN_SIZE + TEST_SIZE), shuffle=True, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=TEST_SIZE)

# Get shape
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# print some sample
print(X_train[:10], y_train[:10])
print(X_test[:10], y_test[:10])

network_arch = [
    {"type": "linear", "input_dim": 2, "output_dim": 10}, 
    {"type": "relu", "input_dim": 10, "output_dim": 10}, 
    {"type": "linear", "input_dim": 10, "output_dim": 20}, 
    {"type": "sigmoid", "input_dim": 20, "output_dim": 20}, 
    {"type": "linear", "input_dim": 20, "output_dim": 10}, 
    {"type": "sigmoid", "input_dim": 10, "output_dim": 10}, 
    {"type": "linear", "input_dim": 10, "output_dim": 2}, 
    {"type": "relu", "input_dim": 2, "output_dim": 2}, 
]

network = NeuralNetwork(network_arch)

