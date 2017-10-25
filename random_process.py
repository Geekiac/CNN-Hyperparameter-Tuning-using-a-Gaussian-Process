# random_process.py
# May 2017 - Steven Smith
#
# Methods for using a random numbers to generate candidate CNN models
#
import cnn
import numpy as np
from os import makedirs, path
from report import output_to_log, plot_figures, log_best_results, log_additional_metrics, log_current_iteration, log_experiment_info, log_current_time
from sklearn.metrics import accuracy_score, log_loss
import time
import traceback

# this is the entry method for a random process experiment
# NOTE: includes error handling to allow other experiments to continue if
#       this experiment fails
def execute(num_iterations = 1, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [0,1]):
    try:
        _execute(num_iterations, max_num_of_epochs, validation_set, numbers_to_test)
    except:
        log_current_time("ExperimentErrors.log", "An ERROR has occurred - Experiment Incomplete!")
        output_to_log("ExperimentErrors.log", traceback.format_exc())

# Performs the actual execution of an experiment
def _execute(num_iterations = 1, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [0,1]):
    start_time = time.time()

    # creates a directory to store logged information and results
    experiment_name = "Random_%si_%se_%sv_%sn" % (num_iterations, max_num_of_epochs, validation_set, len(numbers_to_test))
    results_dir = "c:/results/%s/%s" % (experiment_name, time.strftime("%Y%m%d-%H%M%S"))
    makedirs(results_dir, exist_ok=True)
    log_filename = path.join(results_dir, "log.txt")
    log_experiment_info(log_filename, experiment_name, num_iterations, max_num_of_epochs, validation_set, numbers_to_test)

    # load the train and test datasets
    (trainX, trainY, testX, testY) = cnn.load_data(numbers_to_test)

    # initialise local variables
    params_so_far = np.empty((0, len(cnn.rbf_ranges)))
    accuracy_so_far = np.empty((0,1))
    losses_so_far = np.empty((0,1))

    # Loop to generate new CNN models
    for iteration in range(1, num_iterations + 1):
        # create a random parameter set
        next_params = cnn.random_params()

        log_current_time(log_filename, "Starting Training %r"% (next_params))
        before_train_time = time.time()
        # train the generated model
        model = cnn.train(next_params, trainX, trainY, max_num_of_epochs = max_num_of_epochs, validation_set = validation_set)
        after_train_time = time.time()
        log_current_time(log_filename, "Finished Training %r (%f seconds)" % (next_params, after_train_time - before_train_time))

        # predict the test dataset results using the trained model
        predY = cnn.batch_predict(model, testX, testY)

        # evaluate the models accuracy, log loss, confusion matrix
        # and create a summary report
        predY_1 = predY.argmax(1)
        testY_1 = testY.argmax(1)

        accuracy = accuracy_score(testY_1,predY_1)
        loss = log_loss(testY,predY)

        log_current_iteration(log_filename, iteration, next_params, accuracy, loss)
        log_additional_metrics(log_filename, testY_1, predY_1)

        # keep a running array of each iterations params, accuracy and log loss
        params_so_far = np.concatenate((params_so_far, [next_params]))
        accuracy_so_far = np.append(accuracy_so_far, accuracy)
        losses_so_far = np.append(losses_so_far, loss)

    # the best model has the lowest log loss
    best_iteration = np.argmin(losses_so_far)
    log_best_results(log_filename, best_iteration, params_so_far, accuracy_so_far, losses_so_far)
    plot_figures(results_dir, experiment_name, accuracy_so_far, losses_so_far)

    end_time = time.time()
    output_to_log(log_filename, "Experiment '%s' took %f seconds in total." % (experiment_name, end_time - start_time))
