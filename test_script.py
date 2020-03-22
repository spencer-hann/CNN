import cnn
import numpy as np

from cnn.data import preprocess_data
from cnn.cnn import (
    CNN,
    ConvolutionalLayer,
    MaxPoolingLayer,
    DenseSoftmaxLayer,
    ReLULayer,
)

from matplotlib import pyplot as plt


training_examples, training_targets = \
    preprocess_data("data/mnist_train.csv", max_rows=600)
testing_examples, testing_targets = \
    preprocess_data("data/mnist_test.csv", max_rows=400)

data = (training_examples, training_targets, testing_examples, testing_targets)
train_set = (training_examples, training_targets)
test_set = (testing_examples, testing_targets)


lr = 0.008
cnn = CNN(
    (
        ConvolutionalLayer(1,8,3),
        MaxPoolingLayer(2),
        ConvolutionalLayer(8,3,3),
        #ReLULayer(),
        DenseSoftmaxLayer(14*14*3, 10),
    ),
    lr = lr
)
print(cnn)
print(f"learning rate: {lr}", flush=True)
cnn.train_epochs(6, *train_set)
print("Testing...")
cnn.test(*test_set)

