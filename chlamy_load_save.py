import numpy as np
import os
from motility import *
from scipy.io import loadmat


def load_data(i, max_len, path_in, prefix, file_ending='.txt'):

    xall = np.ma.empty((i, max_len))
    xall.mask = True
    yall = np.ma.empty((i, max_len))
    yall.mask = True
    j = 0
    for file in os.listdir(path_in):
        if file.endswith(file_ending):
            if file.startswith(prefix):
                print(file)
                data = np.loadtxt(path_in + '/' + file, skiprows=1)
                x = data[:, 1]
                y = data[:, 2]
                xall[j, :len(x)] = x
                yall[j, :len(y)] = y
                j += 1

    return xall, yall


def load_data_ordered(i, max_len, path_in, prefix, file_ending='.txt'):

    xall = np.ma.empty((i, max_len))
    xall.mask = True
    yall = np.ma.empty((i, max_len))
    yall.mask = True
    j = 0
    for file in os.listdir(path_in):
        if file.endswith(file_ending):
            if file.startswith(prefix):
                data = np.loadtxt(path_in + '/' + file[0] + str(j + 1) + file[-4:], skiprows=1)
                # print(file[0] + str(j + 1) + file[-4:])
                x = data[:, 1]
                y = data[:, 2]
                xall[j, :len(x)] = x
                yall[j, :len(y)] = y
                j += 1

    return xall, yall






