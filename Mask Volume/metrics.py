# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:30:41 2022

@author: Preshen Naidoo
"""

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff


def compute_Dice_coefficient(original_mask, predicted_mask):
    '''
            2 x Intersection
    DICE = -------------------
           Union + Intersection
           
    This could be seen as:
    
                2 x TP
    DICE = -----------------
           (FP + TP + FN) + TP

    Parameters
    ----------
    original_mask : 2d numpy array
        A binary mask
    predicted_mask : 2d numpy array
        A binary mask

    Returns
    -------
    dc : float
        DIce coefficient value in the range [0,1]

    '''
    
    #a = original_mask.ravel()
    #b = predicted_mask.ravel()
    
    #a = original_mask.flatten()
    #b = predicted_mask.flatten()    
    
    #count1 = (a == 1).sum()
    #count2 = (b == 1).sum()
    
    #count1 = np.count_nonzero(original_mask)
    #count2 = np.count_nonzero(predicted_mask)
    
    a = np.array(original_mask, dtype=np.bool)
    a = np.atleast_1d(a)
    a = tf.reshape(a, [-1])
    
    b = np.array(predicted_mask, dtype=np.bool)
    b = np.atleast_1d(b)
    b = tf.reshape(b, [-1])    
    
    #dice original code, first attempt
    #intersection = np.count_nonzero(a & b)    
    #union = np.count_nonzero(a) + np.count_nonzero(b)
    #dc = (2. * intersection)/float(union) # + intersection - intersection
    
    #dice computation, second attempt using scipy function
    #dc = dice(a, b)
    
    #dice computation, third attempt
    dc = np.sum(predicted_mask[original_mask==1])*2.0 / (np.sum(predicted_mask) + np.sum(original_mask))
    
    #All three above produce the same output. The second attemp needs to be subtracted from 1.
    
    return dc


def compute_IoU(original_mask, predicted_mask):    
    '''
            Intersection
    IoU = -------------------
                Union
           
    This could be seen as:
    
                 TP
    IoU = -----------------
           (FP + TP + FN)

    Parameters
    ----------
    original_mask : 2d numpy array
        A binary mask.
    predicted_mask : 2d numpy array
        A binary mask.

    Returns
    -------
    iou : float
        IoU value.

    '''
    
    a = np.array(original_mask, dtype=np.bool)
    a = np.atleast_1d(a)
    a = tf.reshape(a, [-1])
    
    b = np.array(predicted_mask, dtype=np.bool)
    b = np.atleast_1d(b)
    b = tf.reshape(b, [-1])   
    
    #IoU Attempt 1
    #intersection = np.count_nonzero(a & b)    
    #union = (np.count_nonzero(a) + np.count_nonzero(b)) - intersection      
    #iou = intersection/union
    
    #IoU attempt 2
    intersection = np.sum(predicted_mask[original_mask==1])
    iou = (intersection) / ((np.sum(predicted_mask) + np.sum(original_mask)) - intersection)
    
    #Attempt 1 and 2 above produce the same output. i.e. both works.
    
    return iou



def compute_Hausdorff_distance(original_mask, predicted_mask):
    '''
    We're going to use the Distance Map(Distance Transform) method of computing the Hausdorf distance.
    A python library Scipy has a function that can do this for us called distance_transform_edt.
    After computing the distance map for mask2 i.e DM2, we need to overlap the boundary of mask1 onto 
    DM2. The take the maximum value in DM2 where mask1 overlaps. 
    This maximum value will be the Hausdorf distance d(mask1, mask2).
    Then we need to find the Hausdorf distance d(mask2, mask1).
    The final distance will be the max(d(mask1, mask2), d(mask2, mask1))
    
    However, first lets try to do this using scipy.spatial.distance.directed_hausdorf
    
    Note: for distance, should we use Euclidean or Manhattan etc?
    
    https://en.wikipedia.org/wiki/Distance_transform
    
    https://cs.stackexchange.com/questions/117989/hausdorff-distance-between-two-binary-images-according-to-distance-maps
    
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.morphology.distance_transform_edt.html
    
    Paper: AN IMAGE ALGORITHM FOR COMPUTING THE SDORFF DISTANCE EFFICIENTLY IN LINEAR TIME
    Paper: A Linear Time Algorithm of Computing Hausdorff Distance for Content-based Image Analysis    
    
    '''
    
    hd1 = directed_hausdorff(original_mask, predicted_mask)[0]
    hd2 = directed_hausdorff(predicted_mask, original_mask)[0]
    
    return max(hd1, hd2)    
    #return 0