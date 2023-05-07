#
#   ML NN Template
#   Use Keras and TensorFlow
#
#   Created by Vasili Makouski
#
#   Copyleft
#

import numpy as np
import csv

inptsD = np.array([])
otptsD = np.array([])

# for training NN use both files, one line from "training_data_inputs.csv"  for input, one line from "training_data_results.csv" correct result.

def traindat():
    inptsD = np.genfromtxt('training_data_inputs.csv',delimiter=',', dtype=int)
    otptsD = np.genfromtxt('training_data_results.csv',delimiter=',', dtype=int)
    return  inptsD, otptsD
