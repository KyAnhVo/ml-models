from neuralnetwork import NeuralNetwork
from logistic_regression import LogisticRegressor
import csv
import numpy as np
import numpy.typing as npt
import sys

def create_predictor(attribute_count: int, 
                     layer_count: int, 
                     nodes_per_layer: int, 
                     lr: float, 
                     training_dat_count: int, 
                     class_count: int):
    if layer_count == 0:
        return LogisticRegressor(attribute_count, lr)
    else:
        return NeuralNetwork(attribute_count=attribute_count,
                           layer_count=layer_count,
                           perceptron_per_layer_count=nodes_per_layer,
                           lr=lr,
                           training_dat_count=training_dat_count,
                           class_count=class_count)

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
        train_set_size = len(x_train)

        x_test = []
        y_test = []
        next(test_rows)
        for row in test_rows:
            x_test.append([float(s) for s in row[:-1]])
            y_test.append(float(row[-1]))
        test_set_size = len(x_test)

        nn = create_predictor(attribute_count=attribute_count,
                           layer_count=layer_count,
                           nodes_per_layer=n_per_layer,
                           lr=lr,
                           training_dat_count=len(x_train),
                           class_count=1)

        # train
        for epoch in range(max_epoch):
            index = epoch % train_set_size

            fp_output = nn.train_one(x_train[index], y_train[index])[0]
            
            sum_sqr_err_train = 0
            for i in range(train_set_size):
                prediction = nn.predict(x_train[i])[0]
                sum_sqr_err_train += (prediction - y_train[i]) ** 2
            avr_sqr_err_train = sum_sqr_err_train / train_set_size
            
            sum_sqr_err_test = 0
            for i in range(test_set_size):
                prediction = nn.predict(x_test[i])[0]
                sum_sqr_err_test += (prediction - y_test[i]) ** 2
            avr_sqr_err_test = sum_sqr_err_test / test_set_size

            print(f"In iteration {epoch + 1}: ")
            print(f"Forward pass output: {fp_output:.4f} ")
            print(f"Average squared error on training set ({train_set_size} instances): {avr_sqr_err_train:.4f} ")
            print(f"Average squared error on test set ({test_set_size} instances): {avr_sqr_err_test:.4f} ")
            print()


if __name__ == "__main__":
    main()
