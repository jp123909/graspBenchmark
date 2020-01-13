#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:24:12 2020

@author: jiang
"""


# Script for post-processing and visualizing suction-based grasping
# affordance predictions

import numpy as np
import cv2
from skimage import filters
from pyPostprocess import postprocess
import h5py
from numpy import genfromtxt
import matplotlib.pyplot as plt

# User options (change me)
backgroundColorImage = 'demo/test-background.color.png'   # 24-bit RGB PNG
backgroundDepthImage = 'demo/test-background.depth.png'   # 16-bit PNG depth in deci-millimeters
inputColorImage = 'demo/test-image.color.png'             # 24-bit RGB PNG
inputDepthImage = 'demo/test-image.depth.png'             # 16-bit PNG depth in deci-millimeters
cameraIntrinsicsFile = 'demo/test-camera-intrinsics.csv'  # 3x3 camera intrinsics matrix
resultsFile = 'demo/results.h5'                           # HDF5 ConvNet output file from running infer.lua

# Read RGB-D images and intrinsics
backgroundColor = cv2.imread(backgroundColorImage, cv2.IMREAD_UNCHANGED).astype(np.float)/255
backgroundDepth = cv2.imread(backgroundDepthImage, cv2.IMREAD_UNCHANGED).astype(np.float)/10000
inputColor = cv2.imread(inputColorImage, cv2.IMREAD_UNCHANGED).astype(np.float)/255
inputDepth = cv2.imread(inputDepthImage, cv2.IMREAD_UNCHANGED).astype(np.float)/10000

# load camera intrinstic matrix
cameraIntrinsics = genfromtxt(cameraIntrinsicsFile, delimiter=',')

# Read raw affordance predictions
with h5py.File(resultsFile, 'r') as f:
    # Get the data
    results = np.array(list(f['results'])).astype(np.float)[0]

# swap axes
results = np.swapaxes(results, 0, 2)
results = np.swapaxes(results, 0, 1)

affordanceMap = results[:,:,1] # 2nd channel contains positive affordance
size = (inputDepth.shape[1], inputDepth.shape[0])
affordanceMap = cv2.resize(affordanceMap, size, interpolation = cv2.INTER_LINEAR) # Resize output to full image size 

# Clamp affordances back to range [0,1] (after interpolation from resizing)
affordanceMap[affordanceMap >= 1] = 0.9999
affordanceMap[affordanceMap < 0]= 0

# Post-process affordance predictions and generate surface normals
affordanceMap, surfaceNormalsMap = postprocess(affordanceMap, inputColor,inputDepth, backgroundColor,backgroundDepth, cameraIntrinsics)

# Gaussian smooth affordances
affordanceMap = filters.gaussian(affordanceMap, sigma=7, mode = 'nearest',truncate=2.0) #affordanceMap = imgaussfilt(affordanceMap, 7)

plt.figure()
plt.imshow(inputColor)
plt.imshow(affordanceMap,interpolation='nearest',vmin=0,vmax=1.,cmap='jet', alpha=0.5)
plt.colorbar()
plt.show()
# Generate heat map visualization for affordances
#cmap = jet;
#affordanceMap = cmap(floor(affordanceMap(:).*size(cmap,1))+1,:)
#affordanceMap = reshape(affordanceMap,size(inputColor))

# Overlay affordance heat map over color image and save to results.png
#figure(1); imshow(0.5*inputColor+0.5*affordanceMap)
#figure(2); imshow(surfaceNormalsMap)
#imwrite(0.5*inputColor+0.5*affordanceMap,'results.png')
#imwrite(surfaceNormalsMap,'normals.png')