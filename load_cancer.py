import numpy as np
# import mat73
import os
from scipy.io import loadmat


def load_data_old2(ntrj, lm, path, filename, ending='.txt', cut=np.array([np.inf]), cu=False):
    """
    load all tarjectories from circular tracks
    first angle value is always set to 0
    ignore trajectories with index in cut
    
    """
    if cu:
        lcut=len(cut)
    else:
        lcut=0
    xall=np.ma.empty((ntrj-lcut,lm))
    xall.mask=True    
    
    ltrj=np.zeros(ntrj-lcut)
    j=0
    k=0
    for file in os.listdir(path):
        if file.endswith(ending):
            if file.startswith(filename):
                if j==cut[k]:  #file.startswith(filename+str(cut[k])):
                    k+=1
                    j+=1
                    if k==lcut:
                        k=0
                else:
                    data_array=np.loadtxt(path+'/'+file)
                    phi=data_array[:,2]
                    x=phi*np.mean(data_array[:,3])

                    ltrj[j]=len(x)
                    xall[j,:len(x)]=x

                    j+=1

    # filter out trajectories containing nan values
    indx_notnan = np.where(np.logical_not(np.isnan(xall).any(axis=1)))[0]

    xall = xall[indx_notnan, :]
    ltrj = ltrj[indx_notnan]

    return xall, ltrj

def load_data_cancer(ntrj, lm, path, filename, ending='.npy', cut=np.array([5224])):
    """
    load all tarjectories and fibronectin densities and lane numbers to masked arrays
    first x value is always set to 0
    cut trajectories with index in cut
    
    """
    lcut = len(cut)
    xall = np.ma.empty((ntrj-lcut,lm))
    xall.mask = True
    yall = np.ma.empty((ntrj-lcut,lm))
    yall.mask = True
    dall = np.ma.empty((ntrj-lcut,lm))
    dall.mask = True
    lall = np.zeros(ntrj-lcut)
    ltrj = np.zeros(ntrj-lcut)

    j = 0
    k = 0
    for file in os.listdir(path):
        if file.endswith(ending):
            if file.startswith(filename):
                if file.startswith(filename + '_trj' + str(cut[k])):
                    k+=1
                    if k == lcut:
                        k=0
                else:
                    print(file)
                    data_array = np.load(path + '/' + file)
                    x = data_array[0,:]
                    x = x-x[0] # set first x-position value to 0
                    y = data_array[1,:] # y-position
                    density = data_array[2,:] # fibronectin density on lane
                    lane = data_array[3,:] # lane id
                    ltrj[j] = len(x)

                    xall[j,:len(x)] = x
                    yall[j,:len(y)] = y
                    dall[j,:len(density)] = density
                    lall[j] = lane[1]
                    ltrj[j] = len(x)
                    if np.sum(np.diff(lane)) != 0:
                        print('Lane is not always the same for trajectory', j, file)
                    j += 1

    # filter out trajectories containing nan values in x-position
    indx_notnan = np.where(np.logical_not(np.isnan(xall).any(axis=1)))[0]

    xall = xall[indx_notnan, :]
    yall = yall[indx_notnan, :]
    dall = dall[indx_notnan, :]
    ltrj = ltrj[indx_notnan]
    lall = lall[indx_notnan]

    return xall, yall, dall, lall, ltrj
    
def load_data_cancer_ordered(ntrj, lm, path, filename, ending='.npy', cut=np.array([5224])):
    """
    load all tarjectories and fibronectin densities and lane numbers to masked arrays
    first x value is always set to 0
    cut trajectories with index in cut
    
    """
    lcut = len(cut)
    xall = np.ma.empty((ntrj-lcut,lm))
    xall.mask = True
    yall = np.ma.empty((ntrj-lcut,lm))
    yall.mask = True
    dall = np.ma.empty((ntrj-lcut,lm))
    dall.mask = True
    lall = np.zeros(ntrj-lcut)
    ltrj = np.zeros(ntrj-lcut)

    j = 0
    k = 0
    for file in os.listdir(path):
        if file.endswith(ending):
            if file.startswith(filename):
                if file.startswith(filename + '_trj' + str(cut[k])):
                    k+=1
                    if k == lcut:
                        k=0
                else:
                    data_array = np.load(path + '/' + file[:49] + str(j) + file[-4:]) # hardcoded length of start string is 49
                    #print(file[:48] + str(j) + file[-4:])
                    x = data_array[0,:]
                    x = x - x[0] # set first x-position value to 0
                    y = data_array[1,:] # y-position
                    density = data_array[2,:] # fibronectin density on lane
                    lane = data_array[3,:] # lane id
                    ltrj[j] = len(x)

                    xall[j,:len(x)] = x
                    yall[j,:len(y)] = y
                    dall[j,:len(density)] = density
                    lall[j] = lane[1]
                    ltrj[j] = len(x)
                    if np.sum(np.diff(lane)) != 0:
                        print('Lane is not always the same for trajectory', j, file)
                    j += 1

    # filter out trajectories containing nan values in x-position
    indx_notnan = np.where(np.logical_not(np.isnan(xall).any(axis=1)))[0]

    xall = xall[indx_notnan, :]
    yall = yall[indx_notnan, :]
    dall = dall[indx_notnan, :]
    ltrj = ltrj[indx_notnan]
    lall = lall[indx_notnan]

    return xall, yall, dall, lall, ltrj
