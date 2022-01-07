################################################################################
#                                                                              #
#   Kernel Density Estimator                                                   #
#   Code written by Dawith Lim                                                 #
#                                                                              #
#   Version 1.0.0.0.0.0                                                        #
#   First written on: 2020/08/27                                               #
#   Last modified: 2022/01/06                                                  #
#                                                                              #
#   Packages used:                                                             #
#                                                                              #
################################################################################

import numpy as np
import math
from matplotlib import pyplot as plt

def kde(random_variable, resolution):
    h = bandwidth()
    n = len(random_variable)
    rv = np.around(random_variable, int(math.log(1 / resolution, 10)))
    print(rv)
    start = min(rv)
    end = max(rv)
    bincount = int((end - start) / resolution)
    abcissa = np.linspace(start, end, bincount + 1)
    ordinate = np.linspace(start, end, bincount + 1)
    integ = 0

    for i in range(len(abcissa)):
        ker = kernel((abcissa[i]-rv)/h,'gaussian')
        ordinate[i] = 1/(n * h) * sum(ker)
        integ += ordinate[i] * resolution

    print(integ)

    return np.array([abcissa, ordinate/integ])

def bandwidth():

    return .08

def kernel(argument, kernel_type):
    if kernel_type.lower() == 'gaussian':
        output = np.exp(-argument**2)

    return output

# EOF
