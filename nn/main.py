from neuralnetwork import NeuralNetwork
import csv
import numpy as np
import numpy.typing as npt
import sys

def main():
    layer_count = int(sys.argv[3])
    n_per_layer = int(sys.argv[4])
    lr          = float(sys.argv[5])
    max_epoch   = int(sys.argv[6])

    with open(sys.argv[1], mode='r') as train_file, open(sys.argv[2], mode='r') as test_file:
        train_rows  = csv.reader(train_file, delimiter='\t')
        test_rows   = csv.reader(test_file, delimiter='\t')

        attribute = next(train_rows)
        attribute_count = len(attribute) - 1

        x_train = []
        y_train = []

        for row in train_rows:
            x_train.append([float(s) for s in row[:-1]])
            y_train.append(float(row[-1]))

        x_test = []
        y_test = []
        next(test_rows)
        for row in test_rows:
            x_test.append([float(s) for s in row[:-1]])
            y_test.append(float(row[-1]))

        nn = NeuralNetwork(attribute_count=attribute_count,
                           layer_count=layer_count,
                           perceptron_per_layer_count=n_per_layer,
                           lr=lr,
                           training_dat_count=len(x_train),
                           class_count=1)
        
        # train
        for epoch in range(max_epoch):
            index = epoch % len(x_train)
            prediction = nn.predict(x_train[index])[0]
            if epoch % 100 == 0:
                print(prediction)
            nn.backward(np.array([y_test[index]], dtype=np.float32))


if __name__ == "__main__":
    main()
