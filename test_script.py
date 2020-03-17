import cnn
import numpy as np

from cnn.cnn import ConvolutionalLayer, MaxPoolingLayer
from cnn.data import preprocess_data
from matplotlib import pyplot as plt


dc = "\n\t".join(dir(cnn))
print(f"dir(cnn): {dc}")

training_examples, training_targets = \
    preprocess_data("data/mnist_train.csv", max_rows=100)
testing_examples, testing_targets = \
    preprocess_data("data/mnist_test.csv", max_rows=100)

data = (training_examples, training_targets, testing_examples, testing_targets)
test_set = (testing_examples, testing_targets)

print(f"train.shape: {training_examples.shape}")
img = training_examples[0,1:]
print(f"img.shape: {img.shape}")
img = img.reshape((28,28))
print(f"img.shape: {img.shape}")
print(img)

plt.imshow(img)
plt.savefig(f"{training_targets[0]}.png")
plt.cla()

img = img[np.newaxis, ...]
print(f"img.shape: {img.shape}")

shape1 = (2,1,3)
shape2 = (3,2,9)
print(f"shape1: {shape1}")
print(f"shape2: {shape2}")
conv1 = ConvolutionalLayer(*shape1)
conv2 = ConvolutionalLayer(*shape2)

out = conv1.forward(img)
print(f"conv1 out: {out.shape}")
out = conv2.forward(out)
print(f"conv2 out: {out.shape}")


pool = MaxPoolingLayer(2)
out = pool.forward(img)
print(f"pool2 out: {out.shape}")

plt.imshow(out[0])
plt.savefig(f"after{training_targets[0]}.png")
plt.cla()

