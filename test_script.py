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


zero_mean = True
training_examples, training_targets = preprocess_data(
    "data/mnist_train.csv",
    max_rows=80000,
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


cnn = CNN(
    (
        ConvolutionalLayer(1,16,3),
        MaxPoolingLayer(2),
        #DenseSoftmaxLayer(14*14*4, 10),
        ReLULayer(),
        ConvolutionalLayer(16,32,3),
        MaxPoolingLayer(2),
        ReLULayer(),
        DenseSoftmaxLayer(7*7*32, 10),
    ),
    lr = 0.01,
    lr_decay = 0.8,
)
print(cnn)
cnn.train_test_cycle(3, 2, train_set, test_set, sample_size=400)

#cnn.layers[0].peek_filters()

