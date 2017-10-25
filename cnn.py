# cnn.py
# May 2017 - Steven Smith
#
# Methods for creating and training convolutional neural networks
#
import numpy as np
import tflearn
import tensorflow as tf
import tflearn.datasets.mnist as mnist
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import os
import pdb

# The ranges of each of the parameters in the parameter array
# - comments after each tuple range describe how the range is translated
#   into it's real value as part of the model.
rbf_ranges = [(1,3), # number_of_conv_layers : n = 1, 2, 3
              (1,4), # nb_filter : 2**(n+3) = 16, 32, 64, 128
              (3,10), # filter_size : n = 3, 4, 5, 6, 7, 8, 9, 10
              (1,5), # strides : n = 1, 2, 3, 4, 5
              (1,2), # regularizer : Ln = L1, L2
              (0,1), # use_local_response_normalization : 0 = off or 1 = on
              (0,1), # use_max_pooling : 0 = off or 1 = on
              (2,4), # max_pool_kernel size : n = 2, 3, 4
              (1,3), # number_of_fully_connected_layers : n = 1, 2, 3
              (1,3), # number_of_units : 2**(n+6) = 128, 256, 512
              (1,5), # keep_prob : (0.1 * n) + 0.4 = 0.5, 0.6, 0.7, 0.8, 0.9
              (1,3) # learning_rate : 10**(-n) = 0.1, 0.01, 0.001
             ]

# Creates the convolutional network
def create_cnn(data):
    # Extract parameters from the parameter array - data
    (number_of_conv_layers, nb_filter, filter_size, strides,
    regularizer, use_local_response_normalization, use_max_pooling,
    max_pool_kernel_size, number_of_fully_connected_layers,
    number_of_units, keep_prob, learning_rate) = data

    # input layer
    cnn = input_data(shape=[None,28,28,1], name='input')

    # convolutional layers
    for layer in range(1,number_of_conv_layers+1):
        # convolutional layer with regularization
        cnn = conv_2d(cnn, nb_filter=(2**(nb_filter+3)), filter_size=filter_size,
                    strides=strides, activation='relu', regularizer="L%d" % (regularizer))
        if use_max_pooling == 1:
            # max pooling layer
            cnn = max_pool_2d(cnn, kernel_size=max_pool_kernel_size)
        if use_local_response_normalization == 1:
            # normalization layer
            cnn = local_response_normalization(cnn)

    # fully connected layers with drop out
    for layer in range(1,number_of_fully_connected_layers+1):
        cnn = fully_connected(cnn, n_units=number_of_units, activation='tanh')
        cnn = dropout(cnn, keep_prob=(0.1 * keep_prob) + 0.4)

    # output layer
    cnn = fully_connected(cnn, n_units=10, activation='softmax')

    # regression layer
    cnn = regression(cnn, optimizer='RMSProp', learning_rate=10**(-learning_rate),
                         loss='categorical_crossentropy', name='target')

    return cnn

# trains a convolutional neural network and returns the trained model
def train(next_params, trainX, trainY, max_num_of_epochs = 5, validation_set = 0.7):
    tf.reset_default_graph()
    model = tflearn.DNN(create_cnn(data=next_params.tolist()), tensorboard_verbose=0)

    # the fit splits the trainX and trainY into train and validation sets
    # the datasets are shuffled and the model is trained in batch sizes of 64
    model.fit({'input': trainX}, {'target': trainY}, n_epoch=max_num_of_epochs,
           validation_set=validation_set,
           shuffle=True,
           snapshot_step=64, show_metric=False,
           batch_size=64,run_id='convnet_mnist_new')

    return model

# breaks prediction into batches and joins the predictions back into
# a single prediction array
# NOTE: This gets around out of memory issues predicting against the entire
#       training set
def batch_predict(model, testX, testY, batch_size = 100):
    predY = np.empty((0, testY.shape[1]))
    # Predict in batches of 100 items
    for i in range(0, testY.shape[0], 100):
        x_batch = testX[i:(i+100)]
        pred_batch = np.array(model.predict(x_batch))
        #pdb.set_trace()  #break into debugger
        predY = np.concatenate((predY, pred_batch))
    return predY

# saves a trained model to disk
def save_model(model, filename):
    # if a model has already been saved with the same name, delete it.
    for f in glob.glob("%s.*" % (filename)):
        os.remove(f)
    model.save(filename)

# creates a random set of parameters to build a CNN from
def random_params():
    return np.array([np.random.randint(r[0], r[1] + 1) for r in rbf_ranges])

# Loads the MNIST data
# Joins the train and validation sets into a single training set (split later)
# Returns training and test sets containing only the numbers_to_test
def load_data(numbers_to_test = [0,1]):
    # grabs the data and shapes it into 28 x 28 from 1 x 784
    # joining the train and validation sets
    all_data = mnist.read_data_sets(one_hot=True)
    all_trainX = np.concatenate([all_data.train.images.reshape(-1, 28, 28, 1),
                                    all_data.validation.images.reshape(-1, 28, 28, 1)])
    all_trainY = np.concatenate([all_data.train.labels,
                                    all_data.validation.labels])

    all_testX = all_data.test.images.reshape(-1, 28, 28, 1)
    all_testY = all_data.test.labels

    # intialise filtered datasets
    trainX = np.empty([0, 28, 28, 1])
    trainY = np.empty([0,10])
    testX = np.empty([0, 28, 28, 1])
    testY = np.empty([0,10])

    # only grab digits that are also in numbers_to_test
    for num in numbers_to_test:
        tY = all_trainY.argmax(1) == num
        trainX = np.concatenate([trainX, all_trainX[tY]])
        trainY = np.concatenate([trainY, all_trainY[tY]])

        tY = all_testY.argmax(1) == num
        testX = np.concatenate([testX, all_testX[tY]])
        testY = np.concatenate([testY, all_testY[tY]])

    (trainX, trainY) = shuffle(trainX, trainY)
    (testX, testY) = shuffle(testX, testY)
    return (trainX, trainY, testX, testY)
