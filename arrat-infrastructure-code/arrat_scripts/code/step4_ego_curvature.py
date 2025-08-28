#!/usr/bin/env python
# python step4_ego_curvature.py --settings /path/to/inputs.json --n 1

import json, os, sys, argparse
import numpy as np
import pyproj
from scipy import interpolate
np.float = float
from lib.lib_transforms import veh2map

inputs_filename = 'inputs.json'

def get_min(input_list):
    # min_value, min_index = get_min(np.abs(np.array(s_time)-item['event_start_time']))
    min_value = np.amin(np.array(input_list))
    return min_value, np.where(input_list == min_value)[0][0]

def spline_curvature(tck, unew):
    dx, dy = interpolate.splev(unew, tck, der=1)
    ddx, ddy = interpolate.splev(unew, tck, der=2)
    K = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))
    return K

def spline_fit(x_raw, y_raw, ds, smoothing=None):
    if smoothing is not None:
        tck, u = interpolate.splprep([x_raw, y_raw], s=smoothing)
    else:
        tck, u = interpolate.splprep([x_raw, y_raw])
    u_fit = np.arange(0, 1+ds, ds)
    out = interpolate.splev(u_fit, tck)
    x_fit, y_fit = out[0], out[1]
    k_fit = spline_curvature(tck, u_fit)
    return x_fit, y_fit, k_fit

def find_nearest(x_raw, y_raw, x_fit, y_fit):
    x_near, y_near, i_near = [], [], []
    for i in range(0, len(x_raw)):
        x, y = x_raw[i], y_raw[i]
        dist_all = np.hypot(x_fit-x, y_fit-y)
        min_value, min_index = get_min(np.abs(np.array(dist_all)))
        x_near.append(x_fit[min_index])
        y_near.append(y_fit[min_index])
        i_near.append(min_index)
    return x_near, y_near, i_near


def curvature_by_segments(xin, yin, Nfit, Npad):

    curvature_coarse = []
    curvature_fine = []

    Nis = 0
    end = False

    while not end:

        Nthis = list(range(Nis, min(Nis+Nfit, len(xin)), 1))

        Nstart = max(0, Nthis[0] - Npad)
        Nend = min(len(xin), Nthis[-1] + Npad)
        Nraw = list(range(Nstart, Nend, 1))

        x_raw, y_raw = xin[Nstart:Nend], yin[Nstart:Nend]
        Nt = len(x_raw)
        ds = 1/(Nt*dx_frame) # n points in spline fit are approx 1 ft apart

        # coarse fit
        x_coarse, y_coarse, k_coarse = spline_fit(x_raw, y_raw, ds, smoothing=(smoothing_coarse**2)*len(x_raw))

        # fine fit
        x_fine, y_fine, k_fine = spline_fit(x_raw, y_raw, ds, smoothing=(smoothing_fine**2)*len(x_raw))

        # nearest coarse
        x_near_coarse, y_near_coarse, i_near = find_nearest(x_raw, y_raw, x_coarse, y_coarse)
        k_near_coarse = []
        for i in range(0, len(i_near)):
            k_near_coarse.append(k_coarse[i_near[i]])

        # nearest fine
        x_near_fine, y_near_fine, i_near = find_nearest(x_raw, y_raw, x_fine, y_fine)
        k_near_fine = []
        for i in range(0, len(i_near)):
            k_near_fine.append(k_fine[i_near[i]])

        Nsel = []
        isel = []
        xsel_coarse, ysel_coarse, ksel_coarse = [], [], []
        xsel_fine, ysel_fine, ksel_fine = [], [], []
        for i in range(0, len(Nthis)):
            isel.append(Nraw.index(Nthis[i]))
            Nsel.append(Nraw[Nraw.index(Nthis[i])])
            ksel_coarse.append(k_near_coarse[Nraw.index(Nthis[i])])
            ksel_fine.append(k_near_fine[Nraw.index(Nthis[i])])

        curvature_coarse.extend(ksel_coarse)
        curvature_fine.extend(ksel_fine)

        Nis = Nthis[-1] + 1

        if Nis >= len(xin):
            end = True
    
    return curvature_coarse, curvature_fine

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
marker_base_link = inputs['marker_base_link'] 
map_filename = inputs['map_filename']
dx_frame = inputs['path_spacing']
smoothing_coarse = inputs['smoothing_coarse']
smoothing_fine = inputs['smoothing_fine']

segment_length = 0.25 # miles
mile_to_meter = 1609.34

# Frames file
framesfile = os.path.join(unitdir, framesfile)
f = open(framesfile)
frames_json = json.load(f)
f.close()

# Raw lane line lat, lon
left_line_lat, left_line_lon = [], []
right_line_lat, right_line_lon = [], []

for i, frame_dict in enumerate(frames_json['laneline_frames']):
    veh_lat, veh_lon, veh_hdg = frame_dict['latitude'], frame_dict['longitude'], frame_dict['heading']
    proj_string = " ".join(("+proj=tmerc +ellps=WGS84 +lat_0=",str(veh_lat), "+lon_0=",str(veh_lon), "+k=1.0 +x_0=0 +y_0=0 +units=m +no_defs"))
    pp = pyproj.Proj(proj_string)

    if i == 0:
        proj_string = " ".join(("+proj=tmerc +ellps=WGS84 +lat_0=",str(veh_lat), "+lon_0=",str(veh_lon), "+k=1.0 +x_0=0 +y_0=0 +units=m +no_defs"))
        pp_2 = pyproj.Proj(proj_string)
    
    if frame_dict['line_valid_left']:
        offset_left = frame_dict['line_offset_left']
        left_x, left_y = veh2map(0, 0, veh_hdg, marker_base_link, offset_left)
        left_lon, left_lat = pp(left_x, left_y, inverse=True)
        left_line_lat.append(left_lat)
        left_line_lon.append(left_lon)

    if frame_dict['line_valid_right']:
        offset_right = frame_dict['line_offset_right']
        right_x, right_y = veh2map(0, 0, veh_hdg, marker_base_link, offset_right)
        right_lon, right_lat = pp(right_x, right_y, inverse=True)
        right_line_lat.append(right_lat)
        right_line_lon.append(right_lon)

# Filter by spline (coarse and fine)
        
Nfit = round(segment_length * mile_to_meter / dx_frame)
Npad = Nfit

########################################################################## left
left_x_uf, left_y_uf = [], []

for i in range(0, len(left_line_lat)):
    x, y = pp_2(left_line_lon[i], left_line_lat[i])
    left_x_uf.append(x)
    left_y_uf.append(y)

k_coarse_left, k_fine_left = curvature_by_segments(left_x_uf, left_y_uf, Nfit, Npad)

######################################################################## right
right_x_uf, right_y_uf = [], []
for i in range(0, len(right_line_lat)):
    x, y = pp_2(right_line_lon[i], right_line_lat[i])
    right_x_uf.append(x)
    right_y_uf.append(y)

k_coarse_right, k_fine_right = curvature_by_segments(right_x_uf, right_y_uf, Nfit, Npad)


# Write output

ic = 0
for i, frame_dict in enumerate(frames_json['laneline_frames']):
    if frame_dict['line_valid_left']:
        frame_dict['curvature_coarse_left'] = k_coarse_left[ic]
        frame_dict['curvature_fine_left'] = k_fine_left[ic]
        ic += 1
    else:
        frame_dict['curvature_coarse_left'] = 0.0
        frame_dict['curvature_fine_left'] = 0.0

ic = 0
for i, frame_dict in enumerate(frames_json['laneline_frames']):
    if frame_dict['line_valid_right']:
        frame_dict['curvature_coarse_right'] = k_coarse_right[ic]
        frame_dict['curvature_fine_right'] = k_fine_right[ic]
        ic += 1
    else:
        frame_dict['curvature_coarse_right'] = 0.0
        frame_dict['curvature_fine_right'] = 0.0

json_object = json.dumps(frames_json, indent=4)

outfile_path = framesfile
with open(outfile_path, "w") as outfile:
    outfile.write(json_object)
outfile.close()

print("Created", outfile_path)