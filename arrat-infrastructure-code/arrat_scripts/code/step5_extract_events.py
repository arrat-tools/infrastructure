#!/usr/bin/env python
import argparse, json, os, json, sys, argparse, csv
import numpy as np
np.float = float
from lib.lib_event_extraction import moving_median, moving_std, since_thresh, consec_thresh

# python extract_events.py --settings /path/to/inputs.json --n 1

inputs_filename = 'inputs.json'

parser = argparse.ArgumentParser(
    description="This script accepts one filepath and two boolean values."
)
parser.add_argument(
     "--datadir", default="", help="Unit data directory"
)

args = parser.parse_args()

# Input data and settings file
unitdir = args.datadir
inputs_file = os.path.join(unitdir, inputs_filename)
f = open(inputs_file)
inputs = json.load(f)
f.close()

framesfile = inputs['frames_filename']
radius_threshold = inputs['radius_threshold']

out_framesfile = os.path.join(unitdir, framesfile)
f = open(out_framesfile)
frames_json = json.load(f)
f.close()

print("\n****     Processing Unit", os.path.basename(unitdir), "for lane line events", "    ****")

left_score, right_score = [], []
left_length, right_length = [], []

for frame_dict in frames_json['laneline_frames']:
    left_score.append(frame_dict['line_score_left'])
    right_score.append(frame_dict['line_score_right'])
    left_length.append(frame_dict['line_length_left'])
    right_length.append(frame_dict['line_length_right'])
    
    
left_score_event, right_score_event = [], []
left_length_event, right_length_event = [], []

window_size = inputs['moving_window']
length_stdev_threshold = inputs['length_stdev_threshold']
length_age_tol = inputs['length_age_tol']
score_threshold = inputs['score_threshold']
score_age_tol = inputs['score_age_tol']

# moving statistics
# length

left_length_med = moving_median(left_length, window_size=window_size)
left_length_std = moving_std(left_length_med, window_size=window_size)
left_length_since, left_length_state = since_thresh(left_length_std, length_stdev_threshold, length_age_tol)


right_length_med = moving_median(right_length, window_size=window_size)
right_length_std = moving_std(right_length_med, window_size=window_size)
right_length_since, right_length_state = since_thresh(right_length_std, length_stdev_threshold, length_age_tol)

# contrast 
left_score_since, left_score_state = consec_thresh(left_score, score_threshold, score_age_tol)
right_score_since, right_score_state = consec_thresh(right_score, score_threshold, score_age_tol)

for i, frame_dict in enumerate(frames_json['laneline_frames']):
    frame_dict['visual_state_left'] = left_score_state[i]
    frame_dict['visual_state_right'] = right_score_state[i]
    frame_dict['geom_state_left'] = left_length_state[i]
    frame_dict['geom_state_right'] = right_length_state[i]

    # frame_dict['curv_state_left'] = 0
    # frame_dict['curv_state_right'] = 0
    # frame_dict['curvature_fine_left'] = 0.0
    # frame_dict['curvature_fine_right'] = 0.0
    # frame_dict['curvature_coarse_left'] = 0.0
    # frame_dict['curvature_coarse_right'] = 0.0
    

    if abs(frame_dict['curvature_coarse_left']) >= 1/radius_threshold:
        frame_dict['curv_state_left'] = 0
    else:
        frame_dict['curv_state_left'] = 1

    if abs(frame_dict['curvature_coarse_right']) >= 1/radius_threshold:
        frame_dict['curv_state_right'] = 0
    else:
        frame_dict['curv_state_right'] = 1

    all_left_states = []
    all_left_states.append(frame_dict['visual_state_left'] == 1)
    all_left_states.append(frame_dict['geom_state_left'] == 1)
    all_left_states.append(frame_dict['curv_state_left'] == 1)

    all_right_states = []
    all_right_states.append(frame_dict['visual_state_right'] == 1)
    all_right_states.append(frame_dict['geom_state_right'] == 1)
    all_right_states.append(frame_dict['curv_state_right'] == 1)

    if all(all_left_states):
        frame_dict['overall_state_left'] = 1
    else:
        frame_dict['overall_state_left'] = 0

    if all(all_right_states):
        frame_dict['overall_state_right'] = 1
    else:
        frame_dict['overall_state_right'] = 0

                 
print("Done processing line events")
            
json_object = json.dumps(frames_json, indent=4)
outfile_path = out_framesfile
with open(outfile_path, "w") as outfile:
    outfile.write(json_object)
outfile.close()
print("Created", outfile_path)