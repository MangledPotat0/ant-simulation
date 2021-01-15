################################################################################
#                                                                              #
#   Kernel Density Estimator                                                   #
#   Code written by Dawith Lim                                                 #
#                                                                              #
#   Version 1.0.0.0.0.0                                                        #
#   First written on: 2020/08/27                                               #
#   Last modified: 2020/08/27                                                  #
#                                                                              #
#   Packages used:                                                             #
#                                                                              #
################################################################################

import numpy as np
from matplotlib import pyplot as plt

def kde(random_variable, resolution):
    h = bandwidth()
    n = len(random_variable)
    start = min(random_variable)
    end = max(random_variable)

    abcissa = np.linspace(start, end, resolution)
    ordinate = np.linspace(start, end, resolution)
    integ = 0
    res = (end-start)/resolution

    for i in range(len(abcissa)):
        ker = kernel((abcissa[i]-random_variable)/h,'gaussian')
        ordinate[i] = 1/(n * h) * sum(ker)
        integ += ordinate[i] * res

    print(integ)

    return np.array([abcissa, ordinate/integ])

def bandwidth():

    return .08

def kernel(argument, kernel_type):
    if kernel_type.lower() == 'gaussian':
        output = np.exp(-argument**2)

    return output

