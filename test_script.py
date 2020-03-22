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


def train_cycle(cnn, nepochs):
    print("Training...")
    cnn.train_epochs(4, *train_set)
    print("Testing...")
    cnn.test(*test_set)


zero_mean = True
training_examples, training_targets = preprocess_data(
    "data/mnist_train.csv",
    max_rows=1000,
    zero_mean=zero_mean,
)
testing_examples, testing_targets = preprocess_data(
    "data/mnist_test.csv",
    max_rows=400,
    zero_mean=zero_mean,
)

data = (training_examples, training_targets, testing_examples, testing_targets)
train_set = (training_examples, training_targets)
test_set = (testing_examples, testing_targets)


lr = 0.008
mo = 0.1
cnn = CNN(
    (
        ConvolutionalLayer(1,3,3),
        MaxPoolingLayer(2),
        #DenseSoftmaxLayer(14*14*8, 10),
        ConvolutionalLayer(3,1,3),
        ReLULayer(),
        DenseSoftmaxLayer(14*14, 10),
    ),
    lr = lr,
    #momentum = mo,
)
print(f"learning rate:\t{lr}", flush=True)
print(f"momentum:\t{mo}", flush=True)
print(cnn)
for _ in range(4):
    train_cycle(cnn, 8)

