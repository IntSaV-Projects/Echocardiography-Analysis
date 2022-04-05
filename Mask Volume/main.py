# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 12:05:06 2022

@author: Preshen Naidoo

Example showing how to use the maskvolume module
"""

from maskvolume import *
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    mask_example= cv2.imread('example_mask.png')
    vol, poly_points, minmaxline, midpointline, segments = get_mask_volume(mask_example, K=20, is_binary_image = False, use_bottom_midpoint=False)
    print(vol)
    img = annotate_image(mask_example, poly_points, minmaxline, midpointline, segments)
    plt.imshow(img)

if __name__ == "__main__":
    main()