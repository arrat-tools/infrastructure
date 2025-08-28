#!/usr/bin/env python

import cv2
import numpy as np
from scipy import interpolate

def sort_ascending(input_list):
    sorted_indices = [i[0] for i in sorted(enumerate(input_list), key=lambda x:x[1])]
    sorted_list = [input_list[sorted_indices[i]] for i in range(0, len(sorted_indices))]
    return sorted_indices, sorted_list

def birdseye(img, src, dst):
    height, width = img.shape[0], img.shape[1]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_warp = cv2.warpPerspective(img, M, (width, height))
    return img_warp, M

def get_nn_lines(filepath, resize, crop):
    lane_lines = []
    #lane_lines_warp = []
    line_slopes = []
    line_intercepts = []
    # line_probs = []
    with open(filepath) as f:
        txt_lines = f.readlines()
        n_lines = len(txt_lines)
        n_line = 0
        current_line_index = 0
        lane_line_u, lane_line_v = np.empty(0), np.empty(0)
        for txt_line in txt_lines:
            line_index = int(txt_line.strip("\n").split(" ")[0])
            line_u = int(txt_line.strip("\n").split(" ")[1])*crop[2]/resize[0]+crop[0]
            line_v = int(txt_line.strip("\n").split(" ")[2])*crop[3]/resize[1]+crop[1]
            
            if line_index != current_line_index:
                lane_line_uv = np.array([np.asarray([lane_line_u, lane_line_v]).T.astype(np.float32)])
                lane_lines.append(lane_line_uv)
                current_line_index += 1
                # line slope
                line_fit = np.polyfit(lane_line_u, lane_line_v, 1)
                line_slopes.append(line_fit[0])
                line_intercepts.append(line_fit[1])
                lane_line_u, lane_line_v = np.empty(0), np.empty(0)
                lane_line_u = np.append(lane_line_u, line_u)
                lane_line_v = np.append(lane_line_v, line_v) 
            else:
                lane_line_u = np.append(lane_line_u, line_u)
                lane_line_v = np.append(lane_line_v, line_v)  
                
            if n_line == n_lines-1:
                # close last line if more than one point
                if len(lane_line_u) > 1:
                    lane_line_uv = np.array([np.asarray([lane_line_u, lane_line_v]).T.astype(np.float32)])
                    lane_lines.append(lane_line_uv)
                    line_fit = np.polyfit(lane_line_u, lane_line_v, 1)
                    line_slopes.append(line_fit[0])
                    line_intercepts.append(line_fit[1])
            
            n_line += 1

    return lane_lines, line_slopes, line_intercepts

def get_ego_lines(nn_line_slopes):  
    if len(nn_line_slopes) == 0:
        left_right_indices = [-1, -1]
    elif len(nn_line_slopes) == 1:
        if nn_line_slopes[0] <= 0:
            left_right_indices = [0, -1]
        else:
            left_right_indices = [-1, 0]
    else:
        # Pick the correct two lines when more than two lines are detected
        if len(nn_line_slopes) > 2:
            line_indices = sort_line_slopes(nn_line_slopes)
        else: # otherwise just pick all detected
            line_indices = sort_line_slopes(nn_line_slopes)

        # Find left and right
        if nn_line_slopes[line_indices[0]] <= 0:
            left_right_indices = [line_indices[0], line_indices[1]]
        else:
            left_right_indices = [line_indices[1], line_indices[0]]
    return left_right_indices


def get_sorted_ego_lines(input_list, thresh=1.0):
    # sort slopes
    sorted_indices, sorted_list = sort_ascending(input_list)
    # seperate positive and negative values and indices
    positive_vals = [x for i, x in enumerate(sorted_list) if x >= 0 and x >= thresh]
    positive_indices = [sorted_indices[i] for i, x in enumerate(sorted_list) if x >= 0 and x >= thresh]
    negative_vals = [x for i, x in enumerate(sorted_list) if x < 0 and abs(x) >= thresh]
    negative_indices = [sorted_indices[i] for i, x in enumerate(sorted_list) if x < 0 and abs(x) >= thresh]
    # check if sufficent number of values are found and pick the max/min
    # max for right (+ve)
    if len(positive_vals) > 0:
        if len(positive_vals) > 1:
            max_index = positive_indices[positive_vals.index(max(positive_vals))]
        else:
            max_index = positive_indices[0]
    else:
        max_index = -1
    # min for left (-ve)
    if len(negative_vals) > 0:
        if len(negative_vals) > 1:
            min_index = negative_indices[negative_vals.index(min(negative_vals))]
        else:
            min_index = negative_indices[0]
    else:
        min_index = -1
    # validity
    # left
    if min_index != -1:
        left_valid = True
        left_val = input_list[min_index]
    else:
        left_valid = False
        left_val = 0.0
    # right
    if max_index != -1:
        right_valid = True
        right_val = input_list[max_index]
    else:
        right_valid = False
        right_val = 0.0

    return [left_valid, right_valid], [min_index, max_index], [left_val, right_val]


def ego_line_validity(left_right_indices, line_slopes):
	# line validity
	left_line_validity = [False, False]
	right_line_validity = [False, False]
	left_slope = 0.0
	if left_right_indices[0] != -1:
		left_line_validity[0] = True
		if abs(line_slopes[left_right_indices[0]]) > 0.1:
			left_line_validity[1] = True
			left_slope = line_slopes[left_right_indices[0]]
	right_slope = 0.0	
	if left_right_indices[1] != -1:
		right_line_validity[0] = True
		if abs(line_slopes[left_right_indices[1]]) > 0.1:
			right_line_validity[1] = True
			right_slope = line_slopes[left_right_indices[1]]
	return [all(left_line_validity), all(right_line_validity)], [left_slope, right_slope]

def line_warp_fit(lane_line, M, h):
    lane_line_warp = cv2.perspectiveTransform(lane_line, M)
    u, v, uw, vw = [], [], [], []
    for i in range(0, len(lane_line[0])):
        pt = lane_line[0][i]
        ptw = lane_line_warp[0][i]
        u.append(pt[0])
        v.append(pt[1])
        uw.append(int(ptw[0]))
        vw.append(int(ptw[1]))       
    vmin_fit = max(min(vw), 0)
    vmax_fit = min(max(vw), h)
        # Make fit line
    N = abs(int(vmax_fit)-int(vmin_fit))+1
    vfit = np.linspace(int(vmin_fit), int(vmax_fit), N)
    f = interpolate.interp1d(vw, uw)
    ufit = np.int32(f(vfit))
    return ufit, vfit


def get_ego_lines_warp(lane_lines, left_right_indices, ego_valid, M, height):
    ego_lines = []
    ego_lines_warp = []
    left_u, right_u = 0, 0
		
    if ego_valid[0]:
        lane_line = lane_lines[left_right_indices[0]]
        ufit, vfit = line_warp_fit(lane_line, M, height)
        lane_line_warp = np.array([np.asarray([ufit, vfit]).T.astype(np.float32)]) 
        left_u = ufit[-1]
        #left_u1, left_u2 = max(min(ufit)-dU, 0), min(max(ufit)+dU, width) # u1, u2 of left roi 600, 1000
        ego_lines_warp.append(lane_line_warp)
        ego_lines.append(lane_line)
    
    if ego_valid[1]:
        lane_line = lane_lines[left_right_indices[1]]
        ufit, vfit = line_warp_fit(lane_line, M, height)
        lane_line_warp = np.array([np.asarray([ufit, vfit]).T.astype(np.float32)]) 
        right_u = ufit[-1]
        #right_u1, right_u2 = max(min(ufit)-dU, 0), min(max(ufit)+dU, width) # u1, u2 of right roi 1200, 1600
        ego_lines_warp.append(lane_line_warp)
        ego_lines.append(lane_line)

    return ego_lines, ego_lines_warp, left_u, right_u


def score_from_gray(gray, u1,u2,v1,v2, norm_thresh, vscale):
    roi = gray[v1:v2, u1:u2]
    thresh = get_otsu_threshold(roi)
    maxval = np.max(roi)
    minval = max(0, maxval-thresh)
    clipped = np.clip(roi, thresh, maxval).astype(np.uint8)
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_values = []
    contrast_vector = []
    norm_vector = []
    line_count = 0
    line_sum = 0
    total_v = 0
    for i in range(0, roi.shape[0]):
        this_v = i
        this_row = roi[this_v, :]
        this_contrast = max(this_row) - min(this_row)
        contrast_vector.append(this_contrast)
        this_norm = norm[this_v, :]
        if max(this_norm) - min(this_norm) > norm_thresh:
            line_count += 1
            line_sum = line_sum + this_contrast
        norm_vector.append(max(this_norm) - min(this_norm) )
        total_v += 1
        v_values.append(this_v)

    if line_count == 0:
        score = 0.0
    else:
        score = line_sum/line_count
    length = line_count/vscale
    
    return score, length, contrast_vector


def sort_line_slopes(input_list):
    unsorted_list = abs(np.array(input_list))
    sorted_indices = np.argsort(unsorted_list)
    n = len(sorted_indices)
    output_indices = [sorted_indices[n-1], sorted_indices[n-2]]
    return output_indices

def get_otsu_threshold(image):
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


