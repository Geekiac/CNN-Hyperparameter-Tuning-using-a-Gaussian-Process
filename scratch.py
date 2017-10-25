# scratch.py
# May 2017 - Steven Smith
#
# This is just used to test snippets of code
#
import cnn
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
import time
from os import makedirs,path

start_time = time.time()
num_iterations = 1
max_num_of_epochs = 1
validation_set = 0.7

rbf_ranges = [(1,3), # number_of_conv_layers
              (1,4), # nb_filter : 2**(n+3) = 16, 32, 64, 128
              (2,5), # filter_size
              (1,5), # strides,
              (1,2), # regularizer : Ln
              (0,1), # use_local_response_normalization, off or on
              (0,1), # use_max_pooling, off or on
              (2,4), # max_pool_kernel size
              (1,3), # number_of_fully_connected_layers
              (1,3), # number_of_units : 2**(n+6) = 128, 256, 512
              (1,5), # keep_prob : (0.1 * n) + 0.4 = 0.5, 0.6, 0.7, 0.8, 0.9
              (1,2,3) # learning_rate : 10**(-n) = 0.1, 0.01, 0.001
             ]

next_params = np.array([3, # number_of_conv_layers
                        4, # nb_filter : 2**(n+3) = 16, 32, 64, 128
                        5, # filter_size
                        5, # strides,
                        2, # regularizer : Ln
                        1, # use_local_response_normalization, off or on
                        1, # use_max_pooling, off or on
                        2, # max_pool_kernel size
                        3, # number_of_fully_connected_layers
                        3, # number_of_units : 2**(n+6) = 128, 256, 512
                        4, # keep_prob : (0.1 * n) + 0.4 = 0.5, 0.6, 0.7, 0.8, 0.9
                        3 # learning_rate : 10**(-n) = 0.1, 0.01, 0.001
                         ])

(trainX, trainY, testX, testY) = cnn.load_data([0,1])

model = cnn.train(next_params, trainX, trainY, max_num_of_epochs = max_num_of_epochs, validation_set = validation_set)

predY = cnn.batch_predict(model, testX, testY)

testY_1 = testY.argmax(1)

accuracy = accuracy_score(testY_1,predY)
loss = log_loss(testY_1,predY)

print("Params = %r" % (next_params))
print("Accuracy = %f" % (accuracy))
print("Log Loss = %f" % (loss))

end_time = time.time()
print("Experiment took %f seconds in total." % (end_time - start_time))
