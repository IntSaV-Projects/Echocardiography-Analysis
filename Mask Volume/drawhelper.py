# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:05:00 2022

@author: Preshen Naidoo
"""

import cv2
import numpy as np


def get_rgb_image(predicted_mask):

    img_temp_np = np.zeros((512,512,3))


    for i in range(512):
         for j in range(512):
             if predicted_mask[i,j] == 1:
                img_temp_np[i,j] = (int(255),int(255),int(255))
             else:
                img_temp_np[i,j] = (int(0),int(0),int(0))

    img = img_temp_np.astype(np.uint8)
    
    return img


def draw_poly_on_image(img, coordinates, color, use_weight=True):
    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))
    
    weight = 1
    if (use_weight):
        weight = 2
        
    img = cv2.polylines(img, [pts], True, color, thickness=weight)
    return img


def fill_poly_on_image(img, coordinates, color):
    
    pts = np.array(coordinates, np.int32)
    pts = pts.reshape((-1,1,2))
    img = cv2.fillPoly(img, [pts], color)
    return img


def draw_line_on_image(img, x1, y1, x2, y2, color, use_weight=True):
    
    weight = 1
    if (use_weight):
        weight = 2
        
    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=weight)
    return img


def draw_circle_on_image(img, x1, y1, radius, color, use_weight=True):
    
    weight = 1
    if (use_weight):
        weight = 2
        
    img = cv2.circle(img,(int(x1),int(y1)), radius, color, thickness=weight)
    return img


def show_image(img):
    cv2.imshow('hello_world',img)

    #waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    #cv2.waitKey(0)
    # cv2.destroyAllWindows() simply destroys all the windows we created.
    #cv2.destroyAllWindows()
    c = cv2.waitKey()
    if(c>=0):
        return -1
    return 0