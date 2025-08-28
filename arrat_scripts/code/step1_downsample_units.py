#!/usr/bin/env python
import os, json, sys, pyproj, cv2, argparse, csv, glob
import numpy as np
from lib.lib_state_lrs import load_state_lrs
from lib.lib_get_min import get_min
from lib.lib_project_lrs import project_on_lrs
np.float = float

# python step1_downsample_units.py --datadir /path/to/unit/data

use_lrs = False
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
gpsfile = inputs['gps_filename'] 
framesfile = inputs['frames_filename'] 
PATH_SPACING = inputs['path_spacing']
PATH_SPACING_MAX = inputs['path_spacing_max']
img_ext = '.png'

# Read State LRS
if use_lrs:
    lrs_file = inputs['lrs_file']
    state_lat, state_lon, state_lrs, state_nlfid = load_state_lrs(lrs_file)

# Paths
in_gpsfile = os.path.join(unitdir, gpsfile)
out_framesfile = os.path.join(unitdir, framesfile)
imgdir = os.path.join(unitdir, imgdirname)

time_tol = 0.05
print("\nRoad Audit - Process Lane Lines - Downsample Unit")
print("\n****     Processing Unit " + os.path.basename(unitdir) +   "     ****")

# read gps file
with open(in_gpsfile, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    headers = next(reader)
    data = np.array(list(reader))
    timestamps_strings = data[:,headers.index('Time (sec)')]
    latitude = data[:,headers.index('Latitude (deg)')].astype(float)
    longitude = data[:,headers.index('Longitude (deg)')].astype(float)
    heading = data[:,headers.index('Heading (deg)')].astype(float)

timestamps_gps = []
for time_gps in timestamps_strings:
    timestamps_gps.append(float(time_gps))

# read image times from list of images
globlist = glob.glob(imgdir + "/*" + img_ext)
timestamps_imgs = []

for item in globlist:
    timestamp_string = os.path.basename(item).strip(img_ext)
    timestamps_imgs.append(timestamp_string)
timestamps_imgs.sort()

# empty lists
dx = []
dt = []
timestamps_sel = []
latitude_sel = []
longitude_sel = []
heading_sel = []
lrs_sel = []
i_sel = []


frames_json = {}
frames_list_json = []

# loop through image times
i = 0
dt_list = []
while i < len(timestamps_imgs):
    imgtime = float(timestamps_imgs[i])

    output_dict = {}

    min_value, imin = get_min(np.abs(np.array(timestamps_gps)-imgtime))
    gpstime = timestamps_gps[imin]
    lat, lon = latitude[imin], longitude[imin]
    
    if i == 0:

        t_last = imgtime

        proj_string = " ".join(("+proj=tmerc +ellps=WGS84 +lat_0=",str(lat), "+lon_0=",str(lon), "+k=1.0 +x_0=0 +y_0=0 +units=m +no_defs"))
        pp = pyproj.Proj(proj_string)
        xprev, yprev = 0.0, 0.0
        latitude_sel.append(lat)
        longitude_sel.append(lon)
        timestamps_sel.append(imgtime)
        heading_sel.append(heading[imin])
        
        if use_lrs:
            this_lrs = this_lrs, pp_lat, pp_lon, valid_all, ci, n1, n2 = project_on_lrs(lat, lon, state_lat, state_lon, state_lrs)
        else:
            this_lrs = -1.0
        lrs_sel.append(this_lrs)
        i_sel.append(i)

        distance = 0.0
        output_dict['time'] = timestamps_imgs[i]
        output_dict['latitude'] = lat
        output_dict['longitude'] = lon
        output_dict['lrs'] = this_lrs
        output_dict['dist'] = distance
        output_dict['heading'] = heading[imin]        
        output_dict['path_increment'] = 0.0
        output_dict['time_increment'] = 0.0
        frames_list_json.append(output_dict)

    else:

        dt_list.append(imgtime - t_last)
        t_last = imgtime

        x, y = pp(lon, lat)
        this_dx = np.hypot(x-xprev, y-yprev)
        if this_dx >= PATH_SPACING:
            # check if dx is too large
            if this_dx > PATH_SPACING_MAX:
                # check if prev value isn't already used
                if i-1 > i_sel[-1]:
                    i = i-1
                    imgtime = float(timestamps_imgs[i])
                    min_value, imin = get_min(np.abs(np.array(timestamps_gps)-imgtime))
                    gpstime = timestamps_gps[imin]
                    lat, lon = latitude[imin], longitude[imin]
                    x, y = pp(lon, lat)
                    this_dx = np.hypot(x-xprev, y-yprev)

            if use_lrs:
                this_lrs = this_lrs, pp_lat, pp_lon, valid_all, ci, n1, n2 = project_on_lrs(lat, lon, state_lat, state_lon, state_lrs)
            else:
                this_lrs = -1.0
            lrs_sel.append(this_lrs)
            dt_this = imgtime-timestamps_sel[-1]
            proj_string = " ".join(("+proj=tmerc +ellps=WGS84 +lat_0=",str(lat), "+lon_0=",str(lon), "+k=1.0 +x_0=0 +y_0=0 +units=m +no_defs"))
            pp = pyproj.Proj(proj_string)
            xprev, yprev = 0.0, 0.0
            latitude_sel.append(lat)
            longitude_sel.append(lon)
            heading_sel.append(heading[imin])
            timestamps_sel.append(imgtime)
            dx.append(this_dx)
            distance = distance + this_dx
            dt.append(dt_this)
            i_sel.append(i)

            output_dict['time'] = timestamps_imgs[i]
            output_dict['latitude'] = lat
            output_dict['longitude'] = lon
            output_dict['lrs'] = this_lrs
            output_dict['dist'] = distance
            output_dict['heading'] = heading[imin]        
            output_dict['path_increment'] = this_dx
            output_dict['time_increment'] = dt_this
            frames_list_json.append(output_dict)


    i = i + 1


avg_frame_dist = np.mean(dx)

frames_json['laneline_frames'] = frames_list_json
  
print("\nGPS log")
print("Path Spacing (Mean, StdDev, Max, Min):", np.mean(dx), np.std(dx), np.max(dx), np.min(dx))
print("Time Diff. (Mean, StdDev, Max, Min):", round(1000*np.mean(dt)), round(1000*np.std(dt)), round(1000*(np.max(dt))), round(1000*(np.min(dt))))
print("Downsampled timestamps", len(timestamps_sel), "of", len(timestamps_imgs))

# Output file
json_object = json.dumps(frames_json, indent=4)
with open(out_framesfile, "w") as outfile:
    outfile.write(json_object)
outfile.close()

print("Created", out_framesfile)