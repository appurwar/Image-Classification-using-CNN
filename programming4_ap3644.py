'''

Image Classification using Convolution Neural Networks
@Author - Apoorv Purwar (ap3644)

'''

'''
Part 4)
(4) Evaluate the model. Answer the following questions in a comment at the beginning of the file: Is the binary classification task easier or more difficult than classification into 10 categories?
Justify your response.

Ans) The binary classification task is much easier than classification into 10 categories.
The reason for this is pretty obvious, because our model now has to fit the data only into
2 categories and a much larger set of features can be used to make this classification.

For example, it will be much easier to classify an image as that of animal if it matches
the general characterstics of animals (say 2 eyes, 4 legs and 1 tail), rather than digging deep
into the specific features and classifying the animal as a dog or elephant. Similar argument
holds true for vehicles.

It is much easier to classify images into broader categories than sub dividing them into much
smaller ones, hence the binary classification task is much easier.
'''

import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    #Creation of 1 hot array for training and test labels
    ytrain_1hot = np.zeros(500000, dtype=int).reshape(50000,10)
    ytrain_1hot[np.transpose(np.arange(ytrain.size)), np.transpose(ytrain)] = 1

    ytest_1hot = np.zeros(100000, dtype=int).reshape(10000,10)
    ytest_1hot[np.transpose(np.arange(ytest.size)), np.transpose(ytest)] = 1

    # Normalization of data between 0 and 1 from 0 to 255 of RGB values
    xtrain = xtrain/255
    xtest = xtest/255
    return xtrain, ytrain_1hot, xtest, ytest_1hot

# Output of evaluate(), for 20 epoch -
# Training - loss: 1.3229 - acc: 0.5337
# Test     - loss: 1.5151, acc: 0.4664
def build_multilayer_nn():
    nn = Sequential()
    nn.add(Flatten(input_shape=(32,32,3)))
    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)
    output = Dense(units=10, activation="softmax")
    nn.add(output)
    return nn


def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)

# ****  Primary Results  ****
# Output of evaluate(), for 20 epoch -
# Training - loss: 0.6676 - acc: 0.7642
# Test     - loss: 0.8302 - acc: 0.7195

# ****  Secondary Results (Experimentation)  ****
# Output of evaluate(), for 20 epoch and 4*4*32 pooling layers, and dropout as 0.25 and 0.50 respectively-
# Training - loss: 0.8443 - acc: 0.6991
# Test     - loss: 0.8222 - acc: 0.7099
def build_convolution_nn():
    nn = Sequential()
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Flatten(input_shape=(32,32,3)))
    hidden = Dense(units=250, activation="relu")
    nn.add(hidden)
    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)
    output = Dense(units=10, activation="softmax")
    nn.add(output)
    return nn


def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)


def get_binary_cifar10():
    (train, test) = cifar10.load_data()
    (xtrain, ytrain_initial) = train
    (xtest, ytest_initial) = test
    ytrain = np.zeros(50000)

    for i in range(0, 50000):
        ytrain[i] = (1 if ytrain_initial[i][0] > 1
                     and ytrain_initial[i][0] < 8 else 0)
    ytest = np.zeros(10000)
    for i in range(0, 10000):
        ytest[i] = (1 if ytest_initial[i][0] > 1
                    and ytest_initial[i][0] < 8 else 0)

    xtrain = xtrain / 255
    xtest = xtest / 255

    return (xtrain, ytrain, xtest, ytest)


# Output of evaluate(), for 20 epoch -
# Training - loss: 0.1630 - acc: 0.9367
# Test     - loss: 0.1643 - acc: 0.9366
def build_binary_classifier():

    nn = Sequential()

    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))

    nn.add(Flatten())
    hidden = Dense(units=250, activation='relu')
    nn.add(hidden)
    hidden = Dense(units=100, activation='relu')
    nn.add(hidden)
    output = Dense(units=1, activation='sigmoid')
    nn.add(output)

    return nn

def train_binary_classifier(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)


if __name__ == "__main__":

    # Write any code for testing and evaluation in this main section.
