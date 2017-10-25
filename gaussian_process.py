# gaussian_process.py
# May 2017 - Steven Smith
#
# Methods for using a Gaussian Process to generate candidate CNN models
#
import cnn
import numpy as np
from os import makedirs, path
from report import output_to_log, plot_figures, log_best_results, log_additional_metrics, log_current_iteration, log_experiment_info, log_current_time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.extmath import cartesian
import tflearn
import time
import traceback
import pdb

# this is the entry method for a Gaussian process experiment
# NOTE: includes error handling to allow other experiments to continue if
#       this experiment fails
def execute(num_iterations = 1, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [0,1], acquisition_function="EI"):
    try:
        _execute(num_iterations, max_num_of_epochs, validation_set, numbers_to_test, acquisition_function)
    except:
        log_current_time("ExperimentErrors.log", "An ERROR has occurred - Experiment Incomplete!")
        output_to_log("ExperimentErrors.log", traceback.format_exc())

# Performs the actual execution of an experiment
def _execute(num_iterations = 1, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [0,1], acquisition_function="EI"):
    start_time = time.time()

    # creates a directory to store logged information and results
    experiment_name = "Gaussian_%s_%si_%se_%sv_%sn" % (acquisition_function, num_iterations, max_num_of_epochs, validation_set, len(numbers_to_test))
    results_dir = "c:/results/%s/%s" % (experiment_name, time.strftime("%Y%m%d-%H%M%S"))
    makedirs(results_dir, exist_ok=True)
    log_filename = path.join(results_dir, "log.txt")
    log_experiment_info(log_filename, experiment_name, num_iterations, max_num_of_epochs, validation_set, numbers_to_test)

    # need at least 3 iterations for a Gaussian Process - the first two are
    # random and used to seed the Gaussian Process
    if num_iterations < 3:
        output_to_log(log_filename, "num_iterations (%d < 3) and the Gaussian process needs at least two points to predict" % (num_iterations))
    else:
        # load the train and test datasets
        (trainX, trainY, testX, testY) = cnn.load_data(numbers_to_test)

        # create all of the combinations of params which are used by
        # the Gaussian Process to fit and predict
        log_current_time(log_filename, "Starting Calculating Param Choices")
        before_get_param_choices_time = time.time()
        param_choices = get_param_choices()
        after_get_param_choices_time = time.time()
        log_current_time(log_filename, "Finished Calculating Param Choices (shape=%s) (%f seconds)" % (param_choices.shape, after_get_param_choices_time - before_get_param_choices_time))

        # initialise local variables
        params_so_far = np.empty((0, len(cnn.rbf_ranges)))
        accuracy_so_far = np.empty((0,1))
        losses_so_far = np.empty((0,1))

        # kernel = RBF([1] * len(cnn.rbf_ranges), cnn.rbf_ranges) # radial basis function (squared exponential) kernel
        kernel = Matern([1] * len(cnn.rbf_ranges), cnn.rbf_ranges, nu=2.5) # Matern 5/2 kernel
        # Create a Gaussian Process Regressor with a Matern 5/2 kernel
        gp = GaussianProcessRegressor(kernel=kernel)

        # Loop to generate new CNN models
        for iteration in range(1, num_iterations + 1):
            if iteration < 3:
                # create the first two process seed sets of parameters
                next_params = cnn.random_params()
            else:
                # create new parameters using a Gaussian Process
                gp.fit(params_so_far, losses_so_far)
                meanGP, stdGP = gp.predict(param_choices, return_std=True)
                # use the Expected Improvement acquisition function to
                # determine the best parameter set to try next
                ei = get_expected_improvement(best_loss, meanGP, stdGP)
                best_index = ei.argmax()
                next_params = param_choices[best_index]

                if acquisition_function == "EI_RC":
                    # Randomize the convolutional parameters leaving the rest
                    # generated by the Gaussian Process
                    output_to_log(log_filename, "%s: Before randomize %r" % (acquisition_function, next_params))
                    randomize_convolutional_params(next_params)
                    output_to_log(log_filename, "%s: After randomize %r" % (acquisition_function, next_params))
                elif acquisition_function == "EI_RF":
                    # Randomize the fully connected layer parameters leaving
                    # the rest generated by the Gaussian Process
                    output_to_log(log_filename, "%s: Before randomize %r" % (acquisition_function, next_params))
                    randomize_fully_connected_params(next_params)
                    output_to_log(log_filename, "%s: After randomize %r" % (acquisition_function, next_params))

                # ends the search if the parameter set has been generated before
                if array_in_list(next_params, params_so_far):
                     output_to_log(log_filename, "Terminating search: Parameters have already been tried before (%r)" % (next_params))
                     break

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

            best_loss = min(losses_so_far)
            # if the best log loss is 0, we have found the best model
            if best_loss == 0:
                output_to_log(log_filename, "Terminating search: bestLoss == 0")
                break

        # the best model has the lowest log loss
        best_iteration = np.argmin(losses_so_far)
        log_best_results(log_filename, best_iteration, params_so_far, accuracy_so_far, losses_so_far)
        plot_figures(results_dir, experiment_name, accuracy_so_far, losses_so_far)

    end_time = time.time()
    output_to_log(log_filename, "Experiment '%s' took %f seconds in total." % (experiment_name, end_time - start_time))

# create all of the combinations of params which are used by
# the Gaussian Process to fit and predict
def get_param_choices():
    d = []
    for r in cnn.rbf_ranges:
        d.append(list(range(r[0],r[1]+1)))

    param_choices = cartesian(d)
    return param_choices

# calculate the expected improvement acquisition_function
def get_expected_improvement(best, mean, std):
    ei = best - (mean - (1.96 * std))
    ei[ei < 0] = 0
    return ei

# determine if an array is in an array of arrays
def array_in_list(arr, list_of_arrays):
    return next((True for item in list_of_arrays if np.array_equal(item, arr)), False)

# randomizes just the convolutional layer parameters
def randomize_convolutional_params(params):
    for i in range(0, 8):
        params[i] = np.random.randint(cnn.rbf_ranges[i][0], cnn.rbf_ranges[i][1] + 1)

# randomizes just the fully connected layer parameters
def randomize_fully_connected_params(params):
    for i in range(8, 11):
        params[i] = np.random.randint(cnn.rbf_ranges[i][0], cnn.rbf_ranges[i][1] + 1)