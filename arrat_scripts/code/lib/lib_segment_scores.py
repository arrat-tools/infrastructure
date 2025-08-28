#!/usr/bin/env python
import json, os
import numpy as np
np.float = float
from lib.lib_frame_scores import get_line_visibility_bad, get_line_consistency_bad, get_line_detection_bad, get_line_curv_bad, get_line_combined_bad

def segment_laneline_visibility(frame_list):
    for i, frame_dict in enumerate(frame_list):
        this_bad = get_line_visibility_bad(frame_dict)
        if i == 0:
            bad_dist = 0.0
            bad_count = 0
        if this_bad:
            bad_dist = bad_dist + frame_dict['path_increment']
            bad_count += 1
    return bad_count, bad_dist

def segment_laneline_consistency(frame_list):
    for i, frame_dict in enumerate(frame_list):
        this_bad = get_line_consistency_bad(frame_dict)
        if i == 0:
            bad_dist = 0.0
            bad_count = 0
        if this_bad:
            bad_dist = bad_dist + frame_dict['path_increment']
            bad_count += 1
    return bad_count, bad_dist

def segment_laneline_detection(frame_list):
    for i, frame_dict in enumerate(frame_list):
        this_bad = get_line_detection_bad(frame_dict)
        if i == 0:
            bad_dist = 0.0
            bad_count = 0
        if this_bad:
            bad_dist = bad_dist + frame_dict['path_increment']
            bad_count += 1
    return bad_count, bad_dist

def segment_laneline_combined(frame_list):
    for i, frame_dict in enumerate(frame_list):
        this_bad = get_line_combined_bad(frame_dict)
        if i == 0:
            bad_dist = 0.0
            bad_count = 0
        if this_bad:
            bad_dist = bad_dist + frame_dict['path_increment']
            bad_count += 1
    return bad_count, bad_dist

def segment_laneline_curvature(frame_list):
    for i, frame_dict in enumerate(frame_list):
        this_bad = get_line_curv_bad(frame_dict)
        if i == 0:
            bad_dist = 0.0
            bad_count = 0
        if this_bad:
            bad_dist = bad_dist + frame_dict['path_increment']
            bad_count += 1
    return bad_count, bad_dist

