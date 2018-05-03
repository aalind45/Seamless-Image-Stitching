#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 22:24:27 2018

@author: aalind
"""

import cv2
import numpy as np
a=[]
for num_levels in range(1,7):
    A = cv2.imread("orange.jpg",1)  #Keeping_right
    B = cv2.imread("apple.jpg",1) #Keeping_left
    l = 128
    b = 128
    A = cv2.resize(A,(l,b))
    B = cv2.resize(B,(b,b))
    
    m = np.zeros_like(A, dtype='float32')
    m[:,int(A.shape[1]/2):] = 1 # making the mask 
    
    # generating Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    
    # generate Laplacian Pyramids for A,B and masks
    lpA  = [gpA[num_levels-1]]
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1]) # also reverse the masks
    
    # blending
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    
    # Reconstruction
    final = LS[0]
    for i in range(1,num_levels):
        #print(i)
        final = cv2.pyrUp(final)
        final = cv2.add(final, LS[i])
    result = np.concatenate((A, B,final), axis=1)
    a.append(result)
result = a[0]
for i in range(1,num_levels):
    result= np.concatenate((result,a[i]), axis =0)
cv2.imwrite("result.jpg",result)