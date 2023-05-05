# https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/03_numpy_neural_net
# Build dataset
import os
import numpy as np 

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from loguru import logger

from src.nn import NeuralNetwork
from src.cost import CrossEntropyLoss

TRAIN_SIZE, TEST_SIZE = 2000, 400
BATCH_SIZE = 8
NUM_ITER = 1
RANDOM_SEED = 42

X_data, y_label = make_moons((TRAIN_SIZE + TEST_SIZE), shuffle=True, random_state=RANDOM_SEED)
onehot_y_label = [(1, 0) if x==0 else (0, 1) for x in y_label]
onehot_y_label = np.array(onehot_y_label)

X_train, X_test, y_train, y_test = train_test_split(X_data, onehot_y_label, test_size=TEST_SIZE)

# Convert to one hot vector
# Get shape
logger.info("Sample shape: X_train {}, y_train {}, X_test {}, y_test {}"\
    .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# print some sample
logger.info("some sample from train set")
print(X_train[:10], y_train[:10])

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
print(network)
loss_fn = CrossEntropyLoss()

for it in range(NUM_ITER):
    x, y = X_train[it*BATCH_SIZE: (it+1)*BATCH_SIZE], y_train[it*BATCH_SIZE: (it+1)*BATCH_SIZE]
    pred_y = network.forward(x)
    loss = loss_fn.forward(pred_y, y)
    loss_deri = loss_fn.backward(pred_y, y)
    print(loss.shape, loss_deri.shape)
    network.backward(loss_deri)
    network.update_params()
