import cnn

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

