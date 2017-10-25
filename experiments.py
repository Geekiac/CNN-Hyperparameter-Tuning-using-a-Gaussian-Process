# experiments.py
# May 2017 - Steven Smith
#
# This file is used to execute the experiments against 4 model generators :
#
# 1. Random generation of models
# 2. Gaussian Process Generation of models
# 3. Gaussian Process Generation of models with randomization of convolutional parameters
# 4. Gaussian Process Generation of models with randomization of fully connected parameters
#
import gaussian_process
import random_process
import time

start_time = time.time()


# 20 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 4,7
random_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7])
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7], acquisition_function="EI")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7], acquisition_function="EI_RF")
# 50 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 4,7
random_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7])
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7], acquisition_function="EI")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7], acquisition_function="EI_RF")

# 20 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 4,7,9
random_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9])
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9], acquisition_function="EI_RF")
# 50 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 4,7,9
random_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9])
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [4,7,9], acquisition_function="EI_RF")

# 20 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 0,4,7,9
random_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9])
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9], acquisition_function="EI_RF")
# 50 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 0,4,7,9
random_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9])
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,4,7,9], acquisition_function="EI_RF")

# 20 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 0,2,4,7,9
random_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9])
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9], acquisition_function="EI_RF")
# 50 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 0,2,4,7,9
random_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9])
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,2,4,7,9], acquisition_function="EI_RF")

# 20 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 0,1,2,3,4,5,6,7,8,9
random_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9])
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 20, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9], acquisition_function="EI_RF")
# 50 iterations, 20 epochs, 70/30 train/validation split, numbers to classify 0,1,2,3,4,5,6,7,8,9
random_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9])
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9], acquisition_function="EI")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9], acquisition_function="EI_RC")
gaussian_process.execute(num_iterations = 50, max_num_of_epochs = 20, validation_set = 0.3, numbers_to_test = [0,1,2,3,4,5,6,7,8,9], acquisition_function="EI_RF")

# # For play testing and debugging the algorithms
# random_process.execute(num_iterations = 5, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [4,7])
# gaussian_process.execute(num_iterations = 5, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [4,7], acquisition_function="EI")
# gaussian_process.execute(num_iterations = 5, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [4,7], acquisition_function="EI_RC")
# gaussian_process.execute(num_iterations = 5, max_num_of_epochs = 1, validation_set = 0.7, numbers_to_test = [4,7], acquisition_function="EI_RF")


end_time = time.time()
print("All of the experiments took %f seconds in total." % (end_time - start_time))
