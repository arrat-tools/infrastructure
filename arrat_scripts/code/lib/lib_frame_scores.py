#!/usr/bin/env python
import json, os
import numpy as np
np.float = float

mile2meter = 1609.34

def score_to_color(this_score, score_thresh):
    if this_score <= score_thresh[0]:
        this_color = 'yellow'
        name = "good"
        # segment_dict['color'] = 'good'
        # count_good += 1
    elif this_score >= score_thresh[1]:
        this_color = 'red'
        name = "bad"
        # segment_dict['color'] = 'bad'
        # count_bad += 1
    else:
        this_color = 'orange'
        name = "medium"
        # segment_dict['color'] = 'med'
        # count_med += 1
    return this_color, name

def get_line_combined_bad(frame_dict):
	this_left_bad, this_right_bad = False, False
	if frame_dict['visual_state_left'] == 0 or frame_dict['geom_state_left'] == 0:
		this_left_bad = True
	if frame_dict['visual_state_right'] == 0 or frame_dict['geom_state_right'] == 0:
		this_right_bad=  True
	return this_left_bad or this_right_bad

def get_line_detection_bad(frame_dict):
	this_left_bad, this_right_bad = False, False
	if frame_dict['visual_state_left'] == 0 or frame_dict['geom_state_left'] == 0:
		this_left_bad = True
	if frame_dict['visual_state_right'] == 0 or frame_dict['geom_state_right'] == 0:
		this_right_bad=  True
	return this_left_bad or this_right_bad

def get_line_visibility_bad(frame_dict):
	this_left_bad, this_right_bad = False, False
	if frame_dict['visual_state_left'] == 0:
		this_left_bad = True
	if frame_dict['visual_state_right'] == 0:
		this_right_bad=  True
	return this_left_bad or this_right_bad

def get_line_consistency_bad(frame_dict):
	this_left_bad, this_right_bad = False, False
	if frame_dict['geom_state_left'] == 0:
		this_left_bad = True
	if frame_dict['geom_state_right'] == 0:
		this_right_bad=  True
	return this_left_bad or this_right_bad

def get_line_curv_bad(frame_dict):
	this_bad = False
	if frame_dict['curv_state_left'] == 0 or frame_dict['curv_state_right'] == 0:
		this_bad = True
	return this_bad


# def get_line_seg(frame_list, mode=0):
#     for i, frame_dict in enumerate(frame_list):
#         if mode==1: # line visual only
#             this_bad = get_line_visibility_bad(frame_dict)
#         if mode==2: # line geom only
#             this_bad = get_line_consistency_bad(frame_dict)
#         else: # default mode is combined visbiity and consistency
#             this_bad = get_line_bad(frame_dict)
#         if i == 0:
#             bad_dist = 0.0
#             total_dist = 0.0
#             bad_count = 0
#             total_count = 0
#             start_lrs = frame_dict['lrs']
#         if this_bad:
#             bad_dist = bad_dist + frame_dict['path_increment']
#             bad_count += 1
#         total_dist = total_dist + frame_dict['path_increment']
#         total_count += 1
#     end_lrs = frame_dict['lrs']
#     total_lrs = abs(end_lrs - start_lrs)
#     return total_lrs, total_count, bad_count, total_dist, bad_dist

# def get_curv_seg(frame_list):
# 	for i, frame_dict in enumerate(frame_list):
# 		this_bad = False
# 		if frame_dict['curv_state_left'] == 0 or frame_dict['curv_state_right'] == 0:
# 			this_bad = True
# 		if i == 0:
# 			bad_dist = 0.0
# 			total_dist = 0.0
# 			bad_count = 0
# 			total_count = 0
# 			start_lrs = frame_dict['lrs']
# 		if this_bad:
# 			bad_dist = bad_dist + frame_dict['path_increment']
# 			bad_count += 1
# 		total_dist = total_dist + frame_dict['path_increment']
# 		total_count += 1
# 	end_lrs = frame_dict['lrs']
# 	total_lrs = abs(end_lrs - start_lrs)
# 	return total_lrs, total_count, bad_count, total_dist, bad_dist
