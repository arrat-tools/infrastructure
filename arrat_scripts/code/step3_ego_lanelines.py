#!/usr/bin/env python
import sys, argparse, json, os, json, sys, cv2, argparse, glob, shutil, time, subprocess, yaml, math
import numpy as np
np.float = float
from lib.lib_predictor import get_nn_lines, get_sorted_ego_lines, line_warp_fit, score_from_gray
from lib.lib_algos import process_channel, rmse_fit, marker_cluster_length
# python ego_lanelines.py --datadir /path/to/unit/data

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

imgdirname = inputs['img_dirname']
nndirname = inputs['clrnet_dirname']
framesfile = inputs['frames_filename'] 
RMSE_THRESH = inputs['laneline_rmse_threshold'] # 0.15
MAX_MARKER_GAP = inputs['laneline_marker_gap'] # 1.0

# load images (revised to get timestamps from frames.json)
imgdir = os.path.join(unitdir, imgdirname)

# Frames file
out_framesfile = os.path.join(unitdir, framesfile)
f = open(out_framesfile)
frames_json = json.load(f)
f.close()

dir_clrnet = os.path.join(os.path.dirname(imgdir), nndirname)

# load img cal
w, h = inputs['cal_data']['w'], inputs['cal_data']['h']
um, vm = inputs['cal_data']['um'], inputs['cal_data']['vm']
u1, u2, u3, u4 = inputs['cal_data']['u1'], inputs['cal_data']['u2'], inputs['cal_data']['u3'], inputs['cal_data']['u4']
v1, v2, v3, v4 = inputs['cal_data']['v1'], inputs['cal_data']['v2'], inputs['cal_data']['v3'], inputs['cal_data']['v4']
vscale = h/inputs['cal_data']['lane_marker_span']
lane_width = inputs['cal_data']['left_offset'] - inputs['cal_data']['right_offset'] 
uscale = (w - 2*um)/lane_width
dst_u_left = w/2 - uscale*inputs['cal_data']['left_offset']
dst_u_right = w/2 - uscale*inputs['cal_data']['right_offset']
dst = np.float32([[dst_u_left, h-vm], [dst_u_right, h-vm], [dst_u_left, vm], [dst_u_right, vm]])
src = np.float32([[u1,v1], [u2,v2], [u3,v3], [u4,v4]]) # first warp set
M = cv2.getPerspectiveTransform(src, dst)

# crop settings
top, left = inputs['crop_images']['top'], inputs['crop_images']['left']
crop_width, crop_height = inputs['crop_images']['width'], inputs['crop_images']['height']
if crop_width == -1:
	crop_width = w
if crop_height == -1:
	crop_height = h

resize_width, resize_height = inputs['resize_images']['width'], inputs['resize_images']['height']

left_offset_list, right_offset_list = [], []
left_score_list, right_score_list = [], []
left_length_list, right_length_list = [], []
left_valid_list, right_valid_list = [], []

left_line_lat, left_line_lon, right_line_lat, right_line_lon = [], [], [], []

print("\n****     Processing Unit", os.path.basename(unitdir), "for EGO lane lines", "    ****")


total_frames = len(frames_json['laneline_frames'])
i = 1

for frame_dict in frames_json['laneline_frames']:
	timestamp_string = frame_dict['time']
      
	left_offset, right_offset = 0.0, 0.0
	left_score, right_score = 0.0, 0.0
	left_length, right_length = 0.0, 0.0
	left_valid, right_valid = False, False
	left_rmse, right_rmse = 0.0, 0.0
    
	this_time = float(timestamp_string)
	img = cv2.imread(os.path.join(imgdir, timestamp_string+'.png'))
    
	# img warp
	img_warp = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0])) 
    # gray
	img_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2HSV)[:,:,2]

	# channel for new algo
	channel = cv2.GaussianBlur(img_gray, (3, 3), 0)
	Nscale = 4
	channel = cv2.resize(channel, (channel.shape[1]//Nscale, channel.shape[0]//Nscale), interpolation=cv2.INTER_AREA)

	# line calc params
	dU = int(uscale*inputs['marker_width'])
	dV = int(vscale*inputs['cal_data']['lane_marker_span'])
	v2 = img.shape[0]
	v1 = v2 - dV

	# clrnet lines
	clrnet_file = os.path.join(dir_clrnet, timestamp_string+'.txt')
	
	if os.path.exists(clrnet_file):

		lane_lines, line_slopes, line_intercepts = get_nn_lines(clrnet_file, (resize_width,resize_height), (top,left,crop_width,crop_height))

		# ego lines
		ego_valid, ego_indices, ego_slopes = get_sorted_ego_lines(line_slopes, thresh=inputs['ego_slope_threshold'])
		# print(clrnet_file, ego_valid, w, h, crop_width, crop_height)

		if ego_valid[0]:
			left_valid = True
			left_line = lane_lines[ego_indices[0]]
			ufit, vfit = line_warp_fit(left_line, M, inputs['cal_data']['h'])
			left_u = ufit[-1]
			left_offset = (inputs['cal_data']['w']/2-left_u)/uscale
			left_u1, left_u2 = max(min(ufit)-dU, 0), min(max(ufit)+dU, img.shape[1]) # shape 1 width
			left_score, left_length_, left_contrast = score_from_gray(img_gray, left_u1, left_u2, v1, v2, inputs['marker_contrast_threshold'], vscale)
			left_binary, left_canny = process_channel(channel, [left_u1, left_u2], N=Nscale)
			indices_bright = np.where((left_binary >= 255))
			u_, v_ = indices_bright[1], indices_bright[0]
			left_rmse = rmse_fit(u_/(uscale/Nscale), v_/(vscale/Nscale))
			if not math.isnan(left_rmse):
				left_length = marker_cluster_length(u_, v_, vscale, N=Nscale, gap=MAX_MARKER_GAP)
				if left_rmse > RMSE_THRESH:
					left_length = 0.0

		if ego_valid[1]:
			right_valid = True
			right_line = lane_lines[ego_indices[1]]
			ufit, vfit = line_warp_fit(right_line, M, inputs['cal_data']['h'])
			right_u = ufit[-1]		
			right_offset = (inputs['cal_data']['w']/2-right_u)/uscale
			right_u1, right_u2 = max(min(ufit)-dU, 0), min(max(ufit)+dU, img.shape[1]) # shape 1 width
			right_score, right_length_, right_contrast = score_from_gray(img_gray, right_u1, right_u2, v1, v2, inputs['marker_contrast_threshold'], vscale)
			right_binary, right_canny = process_channel(channel, [right_u1, right_u2], N=Nscale)
			indices_bright = np.where((right_binary >= 255))
			u_, v_ = indices_bright[1], indices_bright[0]
			right_rmse = rmse_fit(u_/(uscale/Nscale), v_/(vscale/Nscale))
			if not math.isnan(left_rmse):
				right_length = marker_cluster_length(u_, v_, vscale, N=Nscale, gap=MAX_MARKER_GAP)
				if right_rmse > RMSE_THRESH:
					right_length = 0.0
                  
	# append
	left_valid_list.append(left_valid)
	right_valid_list.append(right_valid)
	left_offset_list.append(left_offset)
	right_offset_list.append(right_offset)
	left_score_list.append(left_score)
	right_score_list.append(right_score)
	left_length_list.append(left_length)
	right_length_list.append(right_length)
      
	frame_dict['line_valid_left'] = left_valid
	frame_dict['line_valid_right'] = right_valid
	frame_dict['line_offset_left'] = left_offset
	frame_dict['line_offset_right'] = right_offset
	frame_dict['line_score_left'] = left_score
	frame_dict['line_score_right'] = right_score
	frame_dict['line_length_left'] = left_length
	frame_dict['line_length_right'] = right_length
      
	if i % 50 == 0:
		print("Processed frame", str(i), "of", str(total_frames))
	
	i += 1

print("Done processing frames", str(total_frames), "of", str(total_frames))
               
json_object = json.dumps(frames_json, indent=4)

outfile_path = out_framesfile
with open(outfile_path, "w") as outfile:
    outfile.write(json_object)
outfile.close()

print("Created", outfile_path)
            
		

