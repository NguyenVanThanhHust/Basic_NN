from sklearn.datasets import make_blobs

def main():
    TRAIN_SIZE, TEST_SIZE = 2000, 400
    NUM_FEATURES = 2
    RANDOM_SEED=42
    X_data, y_data = make_blobs((TRAIN_SIZE + TEST_SIZE), \
        n_features=NUM_FEATURES, \
            centers=2, 
            random_state=RANDOM_SEED)
    
    with open('data.txt', 'w') as handle:
        for i in range(TRAIN_SIZE):
            x = X_data[i]
            for value in x:
                handle.write(str(value))
                handle.write(" ")
            y = y_data[i]
            handle.write(str(y))
            handle.write("\n")

        for i in range(TEST_SIZE):
            x = X_data[i]
            for value in x:
                handle.write(str(value))
                handle.write(" ")
            y = y_data[i]
            handle.write(str(y))
            handle.write("\n")

if __name__ == "__main__":
    main()
