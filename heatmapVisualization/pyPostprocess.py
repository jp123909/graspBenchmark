#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:30:14 2020

@author: jiang
"""
import numpy as np
import pcl
from scipy.ndimage.filters import uniform_filter

def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)

def postprocess(affordanceMap,inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics):
# Post-process affordance maps with background subtraction and removing
# regions with high variance in 3D surface normals
#
# function affordanceMap = postprocess(affordanceMap,inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics)
# Input:
#   affordanceMap      - 480x640 float array of affordance values in range [0,1]
#   inputColor         - 480x640x3 float array of RGB color values scaled to range [0,1]
#   inputDepth         - 480x640 float array of depth values in meters
#   backgroundColor    - 480x640x3 float array of RGB color values scaled to range [0,1]
#   backgroundDepth    - 480x640 float array of depth values in meters
#   cameraIntrinsics   - 3x3 camera intrinsics matrix
# Output:
#   affordanceMap      - 480x640 float array of post-processed affordance values in range [0,1]
#   surfaceNormalsMap  - 480x640x3 float array of surface normals in camera coordinates (meters)

    # Perform background subtraction to get foreground mask
    foregroundMaskColor = abs(inputColor-backgroundColor) < 0.3 #~(sum(abs(inputColor-backgroundColor) < 0.3,3) == 3)
    foregroundMaskColor = ~(foregroundMaskColor.sum(axis=2) == 3)
    foregroundMaskDepth = (backgroundDepth != 0) & (abs(inputDepth-backgroundDepth) > 0.02)
    foregroundMask = (foregroundMaskColor | foregroundMaskDepth)
    
    # Project depth into 3D camera space
    pixX,pixY = np.meshgrid(np.arange(640), np.arange(480)) # [pixX,pixY] = np.meshgrid(1:640,1:480)
    camX = (pixX-cameraIntrinsics[0,2])*inputDepth/cameraIntrinsics[0,0]
    camY = (pixY-cameraIntrinsics[1,2])*inputDepth/cameraIntrinsics[1,1]
    camZ = inputDepth;
    validDepth = np.where(foregroundMask & (camZ != 0)) # only points with valid depth and within foreground mask
    inputPoints = np.stack([camX[validDepth], camY[validDepth], camZ[validDepth]], axis = 1)
    
    # Compute foreground point cloud normals
    foregroundPointcloud = pcl.PointCloud() #pointCloud(inputPoints.T)
    foregroundPointcloud.from_array(inputPoints.astype(np.float32))
    ne = foregroundPointcloud.make_NormalEstimation()
    tree = foregroundPointcloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.05)
    foregroundNormals = ne.compute()
    
    #return foregroundNormals, foregroundNormals
    
    # Project normals back onto image plane
    pixX = np.round(inputPoints[:,0]*cameraIntrinsics[0,0]/inputPoints[:,2]+cameraIntrinsics[0,2]).astype(np.int)
    pixY = np.round(inputPoints[:,1]*cameraIntrinsics[1,1]/inputPoints[:,2]+cameraIntrinsics[1,2]).astype(np.int)
    pixXY = tuple(np.stack([pixY, pixX]))
    
    foregroundNormals = foregroundNormals.to_array()
    
    surfaceNormalsMap = np.zeros(inputColor.shape)
    surfaceNormalsMap[pixXY] = foregroundNormals[:,:3]
    
    
    # Compute standard deviation of local normals
    meanStdNormals = np.mean(window_stdev(surfaceNormalsMap,25), axis = 2)
    normalBasedSuctionScores = 1 - meanStdNormals/np.nanmax(meanStdNormals)
    normalBasedSuctionScores[np.isnan(normalBasedSuctionScores)] = 0

    
    # Set affordance to 0 for regions with high surface normal variance
    affordanceMap[normalBasedSuctionScores < 0.1] = 0
    affordanceMap[~foregroundMask] = 0
    affordanceMap[np.isnan(affordanceMap)] = 0
    
    return affordanceMap, surfaceNormalsMap
    