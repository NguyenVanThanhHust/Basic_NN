import os, sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

sys.path.append("..")
sys.path.append("../..")
from src.layers.linear import Linear

def main():
    TRAIN_SIZE, TEST_SIZE = 2000, 400
    BATCH_SIZE = 16
    NUM_ITER = int(TRAIN_SIZE / BATCH_SIZE) - 1
    RANDOM_SEED = 42

    X_data, y_data = make_blobs((TRAIN_SIZE + TEST_SIZE), n_features=3, centers=2, random_state=RANDOM_SEED)
    import pdb; pdb.set_trace()
    
    colors = ["red", "green"]
    plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap=matplotlib.colors.ListedColormap(colors))
    cb = plt.colorbar()
    plt.savefig("data.jpg")
    plt.clf()
    return 


if __name__ == "__main__":
    main()