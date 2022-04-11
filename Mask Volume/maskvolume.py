# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:24:55 2022

@author: Preshen Naidoo

This module contains functions for computing the volume of a binary mask.
In order to compute a volume from a single image, a major assumption is that 
it is cylindrical.
"""

import cv2
import math
from sympy import Point, Polygon, Line
from drawhelper import *
from skimage.draw import line

def get_p_at_d(p,v,d):
    '''    
    Computes a point at a certain distance along the given vector.
    
    Parameters
    ----------
    p : tuple
        Represents a 2d coordinate, example (x,y).
    v : tuple
        Represents a 2d vector.
    d : int or float
        Represents distance.

    Returns
    -------
    pTemp : tuple
        A new 2d coordinate.

    '''
    
    v2 = (v[0]*d, v[1]*d)
    pTemp = (p[0]+v2[0], p[1]+v2[1])
    return pTemp

def get_norm(v):
    '''
    Normalises the specified vector.

    Parameters
    ----------
    v : tuple
        Represents a 2d vector

    Returns
    -------
    v_norm : tuple
             The normalised vector.

    '''
    
    vd = math.sqrt(v[0]*v[0]+v[1]*v[1])
    v_norm = (v[0]/vd, v[1]/vd)
    return v_norm

def get_dist(p1, p2):
    '''
    Computes the distance between two points.

    Parameters
    ----------
    p1 : tuple
         Represents a 2d coordinate, example (x,y).
    p2 : tuple
         Represents a 2d coordinate, example (x,y).

    Returns
    -------
    d : float
        DIstance between p1 and p2.

    '''
    
    d = math.sqrt(pow(p2[0]-p1[0],2) + pow(p2[1]-p1[1],2))
    return d

def get_mid(p1, p2):
    '''
    Computes the midpoint between two points, p1 and p2.
    
    Parameters
    ----------
    p1 : tuple
         Represents a 2d coordinate, example (x,y).
    p2 : tuple
         Represents a 2d coordinate, example (x,y).

    Returns
    -------
    TYPE
        tuple.
    TYPE
        A new 2d point i.e. the midpoint.

    '''
    
    return ((p1[0]+p2[0])/2.0, (p1[1]+p2[1])/2.0)



def get_mask_volume(mask, K=20, is_binary_image = True):
    '''
    Computes the volume of the mask with the assumption that the 
    the the width of the segments are used as the radius of a cylinder.

    Parameters
    ----------
    mask : numpy array of 2 or 3 dimensions
           This is a binary mask which means the values contained within
           the array are 1 and 0 only.
    K : int, optional
        The number of segments(or cylinders) to divide the mask into. 
        The default is 20. 
        The greater the number the better the volume approximation, however,
        the more computationally expensive it becomes.
    is_binary_image : bool, optional
        if true, then the input mask is expected to be a binary image.
        Otherwise, it is a normal rgb image.
        Example, if you're passing in an np.array from 
        tensorflow.model.predict' for a segmentation binary classification 
        problem, then this would be a binary image or binary mask.
    use_bottom_midpoint : bool, optional
        if this is true, the segments are based on the centerline from the 
        maximum point(top of LV) to the midpoint at the bottom of the LV. 
        If False, then the centerline runs from the max point to the min point.
        The default is True.

    Returns
    -------
    None.

    '''
    
    use_bottom_midpoint = True

    poly_points = None
    midpointline = None
    minmaxline = None
    segments = None
    
    adj_mask = mask.copy()
    
    if is_binary_image:
        if(len(mask.shape) == 3):
            adj_mask = mask[:, :, 0]
    
    mask_rgb = adj_mask
    
    if is_binary_image:
        mask_rgb = get_rgb_image(adj_mask)

    #Get polyline of the mask    
    image_test1_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_test1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_index_area = 0
    largest_area = -1
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > largest_area):
            largest_area = area
            largest_index_area = i


    #Get the shape/polyline of the largest area    
    poly_points = contours[largest_index_area]    
    
    #get the bounds
    box = cv2.boundingRect(poly_points);
    length_vert = (box[1]+box[3]) - box[1]
    length_hori = (box[0]+box[2]) - box[0]
    min_row = box[1] - length_vert
    max_row = box[1] + box[3] + length_vert
    min_col = box[0] - length_hori
    max_col = box[0] + box[2] + length_hori

    #Find the two points furthest apart
    #While looping through all points, Also find the bottom-left most point and the bottom-right most point
    index1 = -1
    index2 = -1
    dr = -999999999999.00
    d_left = 999999999999.00
    d_right = 999999999999.00
    left_index = -1
    right_index = -1
    for i in range(len(poly_points)):
        coord1 = poly_points[i][0]
        x1 = coord1[0]
        y1 = coord1[1]
        for j in range(len(poly_points)):
            if(i==j):
                continue
            coord2 = poly_points[j][0]
            x2 = coord2[0]
            y2 = coord2[1]        

            d = math.sqrt(pow(x2-x1,2) + pow(y2-y1,2))
            if(d>dr):
                dr = d
                index1 = i
                index2 = j
        
        d1 = get_dist(coord1, (min_col, max_row))
        d2 = get_dist(coord1,(max_col, max_row))
        if(d1 < d_left):
            d_left = d1
            left_index = i
        if(d2 < d_right):
            d_right = d2
            right_index = i        

    coord1 = poly_points[index1][0]
    coord2 = poly_points[index2][0]
    
    #Find min and max and get vect between them
    minP = None
    maxP = None

    if(coord1[1] > coord2[1]):
        minP = coord1
        maxP = coord2
    else:
        minP = coord2
        maxP = coord1
        
    minmaxline = [minP, maxP]
        
    #Get the midpoint
    leftMost = poly_points[left_index][0]
    rightMost = poly_points[right_index][0]
    mid_temp = get_mid(leftMost, rightMost)    
    
    #create sympy polygon object
    pts_temp = []
    for i in range(len(poly_points)):
        coord = poly_points[i][0]
        x1 = coord[0]
        y1 = coord[1]
        pt = Point(x1,y1)
        pts_temp.append(pt)
    poly = Polygon(*pts_temp)
    
    #Get vector from max point to mid point
    v1 = (maxP[0] - mid_temp[0], maxP[1] - mid_temp[1])
    vd = math.sqrt(v1[0]*v1[0]+v1[1]*v1[1])
    v1norm = (v1[0]/vd, v1[1]/vd)
    
    #Find the true midpoint     
    p1_temp = get_p_at_d(maxP, v1norm, dr/10.0)
    p2_temp = get_p_at_d(maxP, v1norm, -dr*1.1)    
    int_pts_temp = poly.intersection(Line(p1_temp, p2_temp))    
    true_mid = None
    #There should only be 2 points of intersection
    if(len(int_pts_temp) >= 2):
        if(get_dist(mid_temp, int_pts_temp[0]) < get_dist(mid_temp, int_pts_temp[1])):
            true_mid = int_pts_temp[0]
        else:
            true_mid = int_pts_temp[1]
    else:
        true_mid = mid_temp
    
    midpointline = [true_mid, maxP]
    
    startPt = maxP
    endPt = minP
    if(use_bottom_midpoint):
        endPt = true_mid
    
    #Get vector from start point to end point
    v1 = (endPt[0] - startPt[0], endPt[1] - startPt[1])
    vd = math.sqrt(v1[0]*v1[0]+v1[1]*v1[1])
    v1norm = (v1[0]/vd, v1[1]/vd)      

    #Get step size/distance    
    distance = get_dist(startPt, endPt) 
    h = distance/K    
#     d_test = int(h*float(K))    
#     true_k = K
#     if(d_test>distance):
#         true_k = true_k-1    
    true_k = K

    #Variables for the loop initialisation
    start = (startPt[0], startPt[1])
    start = get_p_at_d(start, v1norm, -h)
    vol = 0
    segments = []    
    start = startPt
    has_intersections = True
    while(has_intersections):        
        width = 0
        
        #Get point along the line starting from startPt and taking a distance of h
        #Get vect between the next point and the previous point    
        end = get_p_at_d(start, (v1norm[0], v1norm[1]), h)        
        vH = (end[0] - start[0], end[1] - start[1])
        start = (end[0], end[1])
        
        #Safety to get out of the loop
        if(get_dist(startPt, end) > distance*1.5):
            has_intersections = False
            break
        
        #Rotate vect by 90 to get the horizontal cutting line
        vPerp = (vH[1],vH[0]*-1.0)
        vPerp_norm = get_norm(vPerp)

        #Extend length of horizontal line to make sure it cuts
        phoriz1 = get_p_at_d(end, vPerp_norm, -dr/2.0)
        phoriz2 = get_p_at_d(end, vPerp_norm, dr/2.0)        
        
        p1 = Point(phoriz1[0], phoriz1[1])
        p2 = Point(phoriz2[0], phoriz2[1])
        int_pts = poly.intersection(Line(p1, p2))        
        
        if(len(int_pts) == 2):
            int1 = int_pts[0]
            int2 = int_pts[1]
            int_pt1 = (float(int1.x),float(int1.y))
            int_pt2 = (float(int2.x),float(int2.y))
            width = get_dist(int_pt1, int_pt2)            
            segments.append((int_pt1,int_pt2)) 
            
        elif (len(int_pts) > 2):                  
            
            for i in range(len(int_pts)-1):                      
                
                coord1 = int_pts[i]
                coord2 = int_pts[i+1]                
                
                int_pt1 = (float(coord1.x),float(coord1.y))
                int_pt2 = (float(coord2.x),float(coord2.y))
                width += get_dist(int_pt1, int_pt2)                
            
            #just append the first and last for now
            p1 = int_pts[0]
            p2 = int_pts[len(int_pts)-1]
            ip1 = (float(p1.x),float(p1.y))
            ip2 = (float(p2.x),float(p2.y))
            segments.append((ip1,ip2)) 
        else:
            #len(int_pts) is zero or one here
            has_intersections = False
            break
            
        vol += math.pi*width*width*h   
    
    return vol, poly_points, minmaxline, midpointline, segments



def get_intersection(mask_rgb, poly_points, line_pt1, line_pt2):
    
    temp_mask_rgb = mask_rgb.copy()    
    
    #red
    temp_mask_rgb = draw_line_on_image(temp_mask_rgb, line_pt1[0], line_pt1[1], line_pt2[0], line_pt2[1], (255,0,0), use_weight=True)
    
    #green
    temp_mask_rgb = fill_poly_on_image(temp_mask_rgb, poly_points, (0,255,0))
    
    #cyan
    temp_mask_rgb = draw_poly_on_image(temp_mask_rgb, poly_points, (255,255,0), use_weight = False)
    temp_mask_rgb = draw_poly_on_image(temp_mask_rgb, poly_points, (255,255,0), use_weight = False)
    
    pt1 = (int(line_pt1[0]), int(line_pt1[1]))
    pt2 = (int(line_pt2[0]), int(line_pt2[1]))
    discrete_line = list(zip(*line(*pt1, *pt2)))    
    
    int_pts = []
    prev_pix = (0,0,0)
    for i in range(len(discrete_line)):
        coord = discrete_line[i]        
        
        if(abs(coord[1])>=temp_mask_rgb.shape[0] or abs(coord[0])>=temp_mask_rgb.shape[1]):
            continue
            
        pix = temp_mask_rgb[int(coord[1]), int(coord[0])]        
        
        if np.array_equal(pix, np.asarray((255,255,0))):
            int_pts.append(coord)
        elif i > 0:
            #check for change in transition            
            if(np.array_equal(prev_pix, np.asarray((255,0,0))) and np.array_equal(pix, np.asarray((0,255,0)))):
                int_pts.append(coord)
            elif(np.array_equal(prev_pix, np.asarray((0,255,0))) and np.array_equal(pix, np.asarray((255,0,0)))):
                int_pts.append(coord)                
        
        prev_pix = pix
        
    return int_pts



def get_mask_volume_quick(mask, K=20, is_binary_image = True):
    '''
    Computes the volume of the mask with the assumption that the 
    the the width of the segments are used as the radius of a cylinder.

    Parameters
    ----------
    mask : numpy array of 2 or 3 dimensions
           This is a binary mask which means the values contained within
           the array are 1 and 0 only.
    K : int, optional
        The number of segments(or cylinders) to divide the mask into. 
        The default is 20. 
        The greater the number the better the volume approximation, however,
        the more computationally expensive it becomes.
    is_binary_image : bool, optional
        if true, then the input mask is expected to be a binary image.
        Otherwise, it is a normal rgb image.
        Example, if you're passing in an np.array from 
        tensorflow.model.predict' for a segmentation binary classification 
        problem, then this would be a binary image or binary mask.
    use_bottom_midpoint : bool, optional
        if this is true, the segments are based on the centerline from the 
        maximum point(top of LV) to the midpoint at the bottom of the LV. 
        If False, then the centerline runs from the max point to the min point.
        The default is True.

    Returns
    -------
    None.

    '''
    
    use_bottom_midpoint = True

    poly_points = None
    midpointline = None
    minmaxline = None
    segments = None
    
    adj_mask = mask.copy()
    
    if is_binary_image:
        if(len(mask.shape) == 3):
            adj_mask = mask[:, :, 0]
    
    mask_rgb = adj_mask
    
    if is_binary_image:
        mask_rgb = get_rgb_image(adj_mask)

    #Get polyline of the mask    
    image_test1_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image_test1_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_index_area = 0
    largest_area = -1
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if (area > largest_area):
            largest_area = area
            largest_index_area = i


    #Get the shape/polyline of the largest area    
    poly_points = contours[largest_index_area]    
    
    #get the bounds
    box = cv2.boundingRect(poly_points);
    length_vert = (box[1]+box[3]) - box[1]
    length_hori = (box[0]+box[2]) - box[0]
    min_row = box[1] - length_vert
    max_row = box[1] + box[3] + length_vert
    min_col = box[0] - length_hori
    max_col = box[0] + box[2] + length_hori

    #Find the two points furthest apart
    #While looping through all points, Also find the bottom-left most point and the bottom-right most point
    index1 = -1
    index2 = -1
    dr = -999999999999.00
    d_left = 999999999999.00
    d_right = 999999999999.00
    left_index = -1
    right_index = -1
    for i in range(len(poly_points)):
        coord1 = poly_points[i][0]
        x1 = coord1[0]
        y1 = coord1[1]
        for j in range(len(poly_points)):
            if(i==j):
                continue
            coord2 = poly_points[j][0]
            x2 = coord2[0]
            y2 = coord2[1]        

            d = math.sqrt(pow(x2-x1,2) + pow(y2-y1,2))
            if(d>dr):
                dr = d
                index1 = i
                index2 = j
        
        d1 = get_dist(coord1, (min_col, max_row))
        d2 = get_dist(coord1,(max_col, max_row))
        if(d1 < d_left):
            d_left = d1
            left_index = i
        if(d2 < d_right):
            d_right = d2
            right_index = i        

    coord1 = poly_points[index1][0]
    coord2 = poly_points[index2][0]
    
    #Find min and max and get vect between them
    minP = None
    maxP = None

    if(coord1[1] > coord2[1]):
        minP = coord1
        maxP = coord2
    else:
        minP = coord2
        maxP = coord1
        
    minmaxline = [minP, maxP]
        
    #Get the midpoint
    leftMost = poly_points[left_index][0]
    rightMost = poly_points[right_index][0]
    mid_temp = get_mid(leftMost, rightMost)    
    
    #create sympy polygon object
    pts_temp = []
    for i in range(len(poly_points)):
        coord = poly_points[i][0]
        x1 = coord[0]
        y1 = coord[1]
        pt = Point(x1,y1)
        pts_temp.append(pt)    
    
    #Get vector from max point to mid point
    v1 = (maxP[0] - mid_temp[0], maxP[1] - mid_temp[1])
    vd = math.sqrt(v1[0]*v1[0]+v1[1]*v1[1])
    v1norm = (v1[0]/vd, v1[1]/vd)    
    
    #Find the true midpoint     
    p1_temp = get_p_at_d(maxP, v1norm, dr)
    p2_temp = get_p_at_d(maxP, v1norm, -dr)    
    
    int_pts_temp = get_intersection(mask_rgb, poly_points, p1_temp, p2_temp)   
    pts_len = len(int_pts_temp)
    true_mid = None
    #There should only be 2 points of intersection
    if(len(int_pts_temp) >= 2):
        if(get_dist(mid_temp, int_pts_temp[0]) < get_dist(mid_temp, int_pts_temp[pts_len-1])):
            true_mid = int_pts_temp[0]
        else:
            true_mid = int_pts_temp[pts_len-1]
    else:
        true_mid = mid_temp
        
    midpointline = [true_mid, maxP]
    
    startPt = maxP
    endPt = minP
    if(use_bottom_midpoint):
        endPt = true_mid
    
    #Get vector from start point to end point
    v1 = (endPt[0] - startPt[0], endPt[1] - startPt[1])
    vd = math.sqrt(v1[0]*v1[0]+v1[1]*v1[1])
    v1norm = (v1[0]/vd, v1[1]/vd)    
    
    #Get step size/distance    
    distance = get_dist(startPt, endPt) 
    h = distance/K    
#     d_test = int(h*float(K))    
#     true_k = K
#     if(d_test>distance):
#         true_k = true_k-1    
    true_k = K

    #Variables for the loop initialisation
    start = (startPt[0], startPt[1])
    start = get_p_at_d(start, v1norm, -h)
    vol = 0
    segments = []    
    start = startPt
    has_intersections = True
    index = -1
    while(has_intersections):        
        width = 0
        index += 1
        
        #Get point along the line starting from startPt and taking a distance of h
        #Get vect between the next point and the previous point        
        end = get_p_at_d(start, (v1norm[0], v1norm[1]), h)        
        vH = (end[0] - start[0], end[1] - start[1])
        start = (end[0], end[1])
        
        #Safety to get out of the loop
        if(get_dist(startPt, end) >= dr*1.1):
            has_intersections = False
            break
        
        #Rotate vect by 90 to get the horizontal cutting line
        vPerp = (vH[1],vH[0]*-1.0)
        vPerp_norm = get_norm(vPerp)

        #Extend length of horizontal line to make sure it cuts
        phoriz1 = get_p_at_d(end, vPerp_norm, -dr)
        phoriz2 = get_p_at_d(end, vPerp_norm, dr)        
        
        p1 = phoriz1
        p2 = phoriz2    
        int_pts = get_intersection(mask_rgb, poly_points, p1, p2)        
        
        if(len(int_pts) == 2):
            int1 = int_pts[0]
            int2 = int_pts[1]
            int_pt1 = (float(int1[0]),float(int1[1]))
            int_pt2 = (float(int2[0]),float(int2[1]))
            width = get_dist(int_pt1, int_pt2)            
            segments.append((int_pt1,int_pt2)) 
            
        elif (len(int_pts) > 2):
            #just append the first and last for now
            p1 = int_pts[0]
            p2 = int_pts[len(int_pts)-1]
            int_pt1 = (float(p1[0]),float(p1[1]))
            int_pt2 = (float(p2[0]),float(p2[1]))
            width += get_dist(int_pt1, int_pt2)            
            segments.append((int_pt1,int_pt2)) 
        else:
            #len(int_pts) is zero or one here            
            has_intersections = False
            break        
        
        vol += math.pi*width*width*h   
        
    return vol, poly_points, minmaxline, midpointline, segments


def annotate_image(mask, polypoints, minmaxline, midpointline, segments, is_binary_image = True):
    '''
    This function plots the information returned by get_mask_volume.
    Used for visual analysis of the segments and centerline.

    Parameters
    ----------
    mask : TYPE
        As returned from get_mask_volume.
    polypoints : TYPE
        As returned from get_mask_volume..
    minmaxline : TYPE
        As returned from get_mask_volume..
    midpointline : TYPE
        As returned from get_mask_volume..
    segments : TYPE
        As returned from get_mask_volume..
     is_binary_image : bool, optional
        If true, then the input mask is expected to be a binary image.
        Otherwise, it is a normal rgb image.

    Returns
    -------
    mask_rgb : TYPE
        DESCRIPTION.

    '''
    
    adj_mask = mask.copy()
    
    if is_binary_image:
        if(len(mask.shape) == 3):
            adj_mask = mask[:, :, 0]
        
    mask_rgb = adj_mask
    
    if is_binary_image:
        mask_rgb = get_rgb_image(adj_mask)
    
    #Draw the shape/polyline of the largest area onto the mask
    if not polypoints is None:        
        mask_rgb = draw_poly_on_image(mask_rgb, polypoints, (255,255,0), False)
    
    #Draw the line between the two farthest points of the polyline.
    if not minmaxline is None:
        minP = minmaxline[0]
        maxP = minmaxline[1]
        mask_rgb = draw_line_on_image(mask_rgb, minP[0], minP[1], maxP[0], maxP[1], (0,255,0), False)
    
    #Draw the line between the highest point and the bottom mid-point.
    if not midpointline is None:
        midP = midpointline[0]
        maxP = midpointline[1]
        mask_rgb = draw_line_on_image(mask_rgb, midP[0], midP[1], maxP[0], maxP[1], (255,0,255), False)
    
    #Draw the lines of the segments that run accross the mask
    if not segments is None:
        for segment in segments:
            int_pt1 = segment[0]
            int_pt2 = segment[1]
            mask_rgb = draw_line_on_image(mask_rgb, int_pt1[0], int_pt1[1], int_pt2[0], int_pt2[1], (255,0,0), use_weight = False)
            mask_rgb = draw_circle_on_image(mask_rgb, int_pt1[0], int_pt1[1], 2, (0,0,255), False)
            mask_rgb = draw_circle_on_image(mask_rgb, int_pt2[0], int_pt2[1], 2, (0,0,255), False)
            
    return mask_rgb