# https://github.com/SkalskiP/ILearnDeepLearning.py/tree/master/01_mysteries_of_neural_networks/03_numpy_neural_net
# Build dataset
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

TRAIN_SIZE, TEST_SIZE = 2000, 400
RANDOM_SEED = 42
