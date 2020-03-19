import cnn
import numpy as np

from cnn.cnn import CNN, ConvolutionalLayer, MaxPoolingLayer, DenseSoftmaxLayer
from cnn.data import preprocess_data
from matplotlib import pyplot as plt



training_examples, training_targets = \
    preprocess_data("data/mnist_train.csv", max_rows=1000)
testing_examples, testing_targets = \
    preprocess_data("data/mnist_test.csv", max_rows=1000)

data = (training_examples, training_targets, testing_examples, testing_targets)
train_set = (training_examples, training_targets)
test_set = (testing_examples, testing_targets)


lr = 0.005
print(f"learning rate: {lr}", flush=True)

cnn = CNN(
    (
        ConvolutionalLayer(1,3,3, first_layer=True),
        ConvolutionalLayer(3,1,3),
        DenseSoftmaxLayer(28*28, 10),
    ),
    lr = 0.006
)
cnn.train_epochs(10, *train_set)
cnn.test(*test_set)


#print(f"train.shape: {training_examples.shape}")
#img = training_examples[0]
#print(f"img.shape: {img.shape}")
#img = img.reshape((28,28))
#print(f"img.shape: {img.shape}")
#
#plt.imshow(img)
#plt.savefig(f"{training_targets[0]}.png")
#plt.cla()
#
#img = img[np.newaxis, ...]
#print(f"img.shape: {img.shape}")
#
#shape1 = (2,1,3)
#shape2 = (3,2,9)
#print(f"shape1: {shape1}")
#print(f"shape2: {shape2}")
#conv1 = ConvolutionalLayer(*shape1)
#conv2 = ConvolutionalLayer(*shape2)
#
#out = conv1.forward(img)
#print(f"conv1 out: {out.shape}")
#out = conv2.forward(out)
#print(f"conv2 out: {out.shape}")
#
#
#pool = MaxPoolingLayer(2)
#out = pool.forward(out)
#print(f"pool2 out: {out.shape}")
#
#shape3 = (1,3,3)
#print(f"shape3: {shape3}")
#conv3 = ConvolutionalLayer(*shape3)
#
#plt.imshow(out[0])
#plt.savefig(f"after{training_targets[0]}.png")
#plt.cla()
#
#
#out = conv3.forward(out)
#print(f"conv3 out: {out.shape}")
#
#final = DenseSoftmaxLayer(14*14, 10)
#out = final.forward(out)
#print(f"final out: {out.shape}")
#print(out)
#
#layers = [conv1, conv2, pool, conv3, final]
#nn = CNN(layers)
#
#ncorrect = 0
#cost = 0
#for i, (img, label) in enumerate(zip(test_set[0], test_set[1])):
#    img = img.reshape((28,28))[np.newaxis,...]
#    res,_,acc,loss = nn.forward(img, label)
#    ncorrect += acc
#    cost += loss
#    print(res,label,'*' if res == label else ' ', end='  ')
#    print(f"Average accuracy: {ncorrect/(i+1)}, avg. cost: {cost/(i+1)}")

