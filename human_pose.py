
# Code to perform human pose without vizualization
# Library

from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
from time import time
from glob import glob
import argparse

import cv2  as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import matplotlib
from func import *
import pylab as plt

parser = argparse.ArgumentParser()


## Read the image
# TODO : put parameters to have input instead of modifying this file
# test_image = '../../../db/track-runner.jpg'
parser.add_argument("input", type=str)
parser.add_argument('output', type=str)
parser.add_argument('--multiple', default=False, action='store_true', help='deal with multiple image')
args = parser.parse_args()
img_paths = []

if(args.multiple):
    img_paths = sorted(glob(join(args.input, '*[0-9].jpg')))
else:
    img_paths.append(args.input) # B,G,R order

    
# get the parameters and the scale search( modify it for foreground)
for ind, img in enumerate(img_paths):
    oriImg = cv.imread(img)
    print('ind:'+str(ind))
    param, model = config_reader()
    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

# Neural net parameter
    if(ind == 0):
        if param['use_gpu']: 
            caffe.set_mode_gpu()
            caffe.set_device(param['GPUdeviceNumber']) # set to your device!
        else:
            caffe.set_mode_cpu()
        net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

    heatmap_avg, paf_avg = get_heatmap(oriImg, param, model, multiplier, net)
    # use neural net to get the links
    all_peaks, peak_counter = NMS(heatmap_avg, paf_avg, param)  # non maximum suppression
    connection_all, special_k = create_connection(all_peaks, peak_counter, paf_avg, oriImg, param)
    # print(len(connection_all))
    #print(all_peaks[0])
    #print(connection_all)
    subset, candidate = create_subset(connection_all, special_k, all_peaks, peak_counter)
    # print(subset)

    list_cand, n_pers = save_person(subset, candidate)
    if(len(list_cand)>0):
        np.savez(args.output+str(ind)+'.npz',list_cand[0])
    else:
        print('no frame')
    
    
