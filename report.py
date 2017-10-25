# report.py
# May 2017 - Steven Smith
#
# Methods for logging results and outputting plots
#
import matplotlib.pyplot as plt
import numpy as np
from os import path
from sklearn.metrics import classification_report, confusion_matrix
import time

# sends text to the experiment log file (and the console)
def output_to_log(log_filename, text):
    with open(log_filename, 'a') as f:
        print(text)
        f.write("%s\n" % (text))

# sends the current time and text to the experiment log file (and the console)
def log_current_time(log_filename, text=''):
    output_to_log(log_filename, "%s: %s" % (time.strftime("Current Time: %d-%m-%Y %H:%M:%S"), text))

# sends experiment information to the experiment log file (and the console)
def log_experiment_info(log_filename, experiment_name, num_iterations, max_num_of_epochs, validation_set, numbers_to_test):
    log_current_time(log_filename)
    output_to_log(log_filename, "experiment_name = %s" % (experiment_name))
    output_to_log(log_filename, "num_iterations = %s" % (num_iterations))
    output_to_log(log_filename, "max_num_of_epochs = %s" % (max_num_of_epochs))
    output_to_log(log_filename, "validation_set = %s" % (validation_set))
    output_to_log(log_filename, "numbers_to_test = %s" % (numbers_to_test))

# sends current iteration to the experiment log file (and the console)
def log_current_iteration(log_filename, iteration, current_params, accuracy, log_loss):
    log_current_time(log_filename)
    output_to_log(log_filename, "Current Iteration = %s" % (iteration))
    output_to_log(log_filename, "Current Params = %r" % (current_params))
    output_to_log(log_filename, "Current Accuracy = %s" % (accuracy))
    output_to_log(log_filename, "Current Log Loss = %s" % (log_loss))

# sends best results to the experiment log file (and the console)
def log_best_results(log_filename, best_iteration, params, accuracies, losses):
    log_current_time(log_filename)
    output_to_log(log_filename, "Best Iteration = %s" % (best_iteration))
    output_to_log(log_filename, "Best Params = %r" % (params[best_iteration]))
    output_to_log(log_filename, "Best Accuracy = %s" % (accuracies[best_iteration]))
    output_to_log(log_filename, "Best Log Loss = %s" % (losses[best_iteration]))
    output_to_log(log_filename, "All params\n%r" % (params))
    output_to_log(log_filename, "All accuracies\n%r" % (accuracies))
    output_to_log(log_filename, "All losses\n%r" % (accuracies))

# sends a classification report and confusion matrix to the experiment
# log file (and the console)
def log_additional_metrics(log_filename, testY, predY):
    log_current_time(log_filename)
    output_to_log(log_filename, "\nClassification Report\n%s\n" % (classification_report(testY, predY)))
    output_to_log(log_filename, "\nConfusion Matrix\n%s\n" % (confusion_matrix(testY, predY)))

# plots the accuracies and the log losses to png images
def plot_figures(results_dir, experiment_name, accuracies, log_losses):
    num_iterations = accuracies.shape[0]
    plt.figure(1)
    plt.plot(np.arange(1, accuracies.shape[0] + 1), accuracies, 'bo')
    plt.axis([0, num_iterations, 0.0, 1.0])
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("%s Accuracy" % (experiment_name))
    plt.savefig(path.join(results_dir, 'accuracy.png'), format='png')
    plt.close()

    plt.figure(2)
    plt.plot(np.arange(1, log_losses.shape[0] + 1), log_losses, 'ro')
    plt.axis([0, num_iterations, 0.0, max(log_losses)])
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")
    plt.title("%s Log loss" % (experiment_name))
    plt.savefig(path.join(results_dir, 'log_loss.png'), format='png')
    plt.close()
