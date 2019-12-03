# Ravishankar, Dheeraj
# 1001-652-847
# 2019-11-27
# Assignment-04-03

import pytest
import numpy as np
from p4.cnn import CNN
import os

def test_evaluate():
    from tensorflow.keras.datasets import mnist
    import tensorflow.keras as keras

    batch_size = 128
    num_classes = 10
    epochs = 3
    image_row, image_column = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = CNN()
    model.add_input_layer(shape=(28, 28, 1), name="Input")
    model.append_conv2d_layer(32, kernel_size=(3, 3))
    model.append_conv2d_layer(64, kernel_size=(3, 3))
    model.append_maxpooling2d_layer(pool_size=(2, 2))
    model.append_flatten_layer()
    model.append_dense_layer(128, activation="relu")
    model.append_dense_layer(num_classes, activation="softmax")
    model.set_loss_function("categorical_crossentropy")
    model.set_metric("accuracy")
    model.set_optimizer("Adagrad")
    model.train(x_train, y_train, batch_size=batch_size, num_epochs=epochs)

    mark = model.evaluate(x_test, y_test)
    true = np.array([0.05093684684933396, 0.9907])

    np.testing.assert_almost_equal(true[0], mark[0], decimal=2)
    np.testing.assert_almost_equal(true[1], mark[1], decimal=2)

def test_train():
    from tensorflow.keras.datasets import mnist
    import tensorflow.keras as keras

    batch_size = 128
    num_classes = 10
    epochs = 1
    image_row, image_column = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = CNN()
    model.add_input_layer(shape=(28, 28, 1), name="Input")
    model.append_conv2d_layer(32, kernel_size=(3, 3))
    model.append_conv2d_layer(64, kernel_size=(3, 3))
    model.append_maxpooling2d_layer(pool_size=(2, 2))
    model.append_flatten_layer()
    model.append_dense_layer(128, activation="relu")
    model.append_dense_layer(num_classes, activation="softmax")
    model.set_loss_function("categorical_crossentropy")
    model.set_metric("accuracy")
    model.set_optimizer("Adagrad")
    model.train(x_train, y_train, batch_size=batch_size, num_epochs=epochs)

    mark = model.evaluate(x_test, y_test)
    true = np.array([0.06997422293154523, 0.9907])

    np.testing.assert_almost_equal(true[0], mark[0], decimal=2)
    np.testing.assert_almost_equal(true[1], mark[1], decimal=2)

