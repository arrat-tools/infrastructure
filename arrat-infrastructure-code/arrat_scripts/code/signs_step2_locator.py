# usage:  python signs_step2_locator.py --datadir /path/to/unit

import os, json, pyproj, argparse, pathlib, shutil
from PIL import Image, ImageDraw, ImageFont
from lib.lib_signs_tools import mk_temp_dir_bc, distance_pt2line, longest_increasing_subsequence_indices_greater_than_constant, lmap
import numpy as np
from lib.lib_latlon_to_en import *
from lib.lib_latlon_from_en import *
from lib.lib_local2global import *
from lib.lib_global2local import *
from lib.lib_uv2xy_func import *

inputs_filename = 'inputs.json'
unit_metafilename = 'unit_metadata.json'

def read_json(file_dir):
    f = open(file_dir)
    file = json.load(f) 
    f.close()
    return file

def sign_geojson(lon, lat, c_name, r_dep, r_x, sign_side, sign_dict, sign_score, leg_time, conspi_ratio, glan_leg, und_max,  score_leg, score_csp, score_gla, score_und, overall_score, text, datadir, geojsonfile):
	geojson_dict = {
        "type": "FeatureCollection",
        "features": []
    }
	for i in range(len(c_name)):
		feature_dict = {
			"type": "Feature",
			"properties": {
                "filename": c_name[i],
                "depth": r_dep[i],
                "roi_x": r_x[i],
                "quality": 100*float(sign_score[i]),
                "legibility_time": leg_time[i],
                "legibility_time_score": score_leg[i],
                "conspicuity": conspi_ratio[i],
                "conspicuity_score": score_csp[i],
                "glance_legibility": glan_leg[i],
                "glance_legibility_score": score_gla[i],
                "understandability": und_max[i],
                "understandability_score": score_und[i],
                "overall_score": overall_score[i],
                "info": text[i],
                "sign_side": sign_side[i],
                "sub_filename": sign_dict[c_name[i]]
			},
			"geometry": {
				"type": "Point", #Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, and GeometryCollection
				"coordinates": [lon[i], lat[i]]
			}         }
		geojson_dict['features'].append(feature_dict)
	json_object = json.dumps(geojson_dict, indent=4)
	with open(os.path.join(datadir, geojsonfile), "w") as outfile:
		outfile.write(json_object)
	outfile.close()
	print("GeoJson File created", geojsonfile)

s1 = 's1_group_det_res.json'
s4 = 's4_group_det_res.json'
s5 = 's5_group_det_res.json'


parser = argparse.ArgumentParser(description="This script accepts one input filepath and unit number to do the sign recognition.")
parser.add_argument("--datadir", default="", help="Unit data directory")
args = parser.parse_args()

# Inputs
unitdir = args.datadir
inputs_file = os.path.join(unitdir, inputs_filename)
f = open(inputs_file)
inputs = json.load(f)
f.close()
# Unit metadata from operator gui
unit_metafile = os.path.join(unitdir, unit_metafilename)
f = open(unit_metafile)
unit_metadata = json.load(f)
f.close()

# User defines
imu_switch = True #parse_boolean(args.hdg) # set to true if heading data is available; set to false to calculate heading using veh positions.
CAM_GPS_LONG = inputs['cal_data']['cam_gps_long'] # distance between camera and gps in meters
VALID_DEPTH_RANGE = [inputs['VALID_DEPTH_RANGE_min'], inputs['VALID_DEPTH_RANGE_max']] # Depth consideed invalid outside of this range in meters, is (0,15)
nominal_sign_depth = inputs['nominal_sign_depth']
nominal_sign_lateral = inputs['nominal_sign_lateral']
print('VALID_DEPTH_RANGE(min, max)', VALID_DEPTH_RANGE)
img_size = (inputs['cal_data']['w'], inputs['cal_data']['h']) # width, height in pixels
ra = inputs['valid_bbox_area']  # percentage for reduced area for the depth calculation. Use 0 for the original area from bbox. range: 0 <= ra < 0.5
print('valid_bbox: ', round(100*(1-2*ra)),'% of the area')
# Camera intrinsics
fx = inputs['cal_data']['fx']
fy = inputs['cal_data']['fy']
cx = inputs['cal_data']['cx']
cy = inputs['cal_data']['cy']
# file path
rootdir = unitdir
geojson_name = inputs['geojson_name']
gpsfile = os.path.join(rootdir,inputs['gps_filename'])
depdir = os.path.join(rootdir, inputs['dep_dir']) #os.path.join(rootdir, 'NN_filtered_depth')
print('Specifying data path - Complete')
# ****************************** Extract GPS data ******************************
gps_file = open(gpsfile, 'r')
gps_lines = gps_file.readlines()
latlon_t, lat, lon, heading_deg, spd = [],[],[],[],[]
n = 0
for line in gps_lines:
    elements = line.split()
    if n == 0:  
        key = elements[0::2] # a list of headers
    else: # The first line of the file is header    
        latlon_t.append(float(elements[key.index('Time')]))
        lat.append(float(elements[key.index('Latitude')]))
        lon.append(float(elements[key.index('Longitude')]))
        heading_deg.append(float(elements[key.index('Heading')]))
        # spd.append(float(elements[key.index('Speed')]))
    n+=1

print('Loading lat/long - Complete')
""" Pyproj """
reflat, reflon = lat[0],lon[0] 
print('GPS starting point: ',reflat, reflon)
proj_string = " ".join(("+proj=tmerc +ellps=WGS84 +lat_0=",str(reflat), "+lon_0=",str(reflon),
                          "+k=1.0 +x_0=",str(0.0), "+y_0=", str(0.0), "+units=m +no_defs"))
pp = pyproj.Proj(proj_string)

# ****************************** MZ's sign scoring pipeline ******************************
RUN_STEP1 = True
COPY_BY_GROUP = True
RUN_STEP2 = True
DRAW_OCR = True
RUN_STEP3 = True
BYPASS_Q_BBOX_AREA = True
VIS_ANCHOR = True
RUN_STEP4 = True
RUN_STEP5 = True
RUN_STEP6 = False

width = inputs['cal_data']['w'] #2208    # image width
group_max_time_diff = 0.5   # second
group_min_frame_count = 2
max_speed_levenshtein_dist = 2
max_limit_levenshtein_dist = 2
distance_threshold = 50 # for bbox to be classified as the same class in tracking
anchor_resolution_deg = 5
anchor_left_max_deg = 200
anchor_left_min_deg = 160
anchor_right_max_deg = 20
anchor_right_min_deg = -20
# step 4
# step 5
camera_capture_freq = 5 # Hz
lmap_u_max = 0.99
lmap_u_min = 0.79
lmap_l_max = 1.0
lmap_l_min = 0.4
lmap_g_max = 1.0
lmap_g_min = 0.9
lmap_c_max = 0.75
lmap_c_min = 0.5
weight_u = 0.25
weight_l = 0.25
weight_g = 0.25
weight_c = 0.25
assert weight_u + weight_l + weight_g + weight_c == 1.0

# ------


detection_result_path = os.path.join(rootdir, inputs['ImageProc_dirname'], inputs['signs_results_filename']) #rootdir+ "/signs_ocr/custom_output.json"
data_path = os.path.join(rootdir, inputs['img_dirname']) 
data_temp = os.path.join(rootdir, "sings_details")
speed_file_path = ""
print("data_root: ", rootdir)
print("detection_result_path: ", detection_result_path)
print("data_path: ", data_path)
s1_dir = os.path.join(data_temp, "s1_group")
s4_dir = os.path.join(data_temp, "s4_stat")
s5_dir = os.path.join(data_temp, "s5_criterion")


def copy_files(src_path, filenames, dst_path, dst_folder=None, img_format="png"):
    if dst_folder:
        dst_path = os.path.join(dst_path, dst_folder)
        pathlib.Path(dst_path).mkdir()
    for fn in filenames:
        shutil.copy(os.path.join(src_path, f'{fn}.{img_format}'), dst_path)

def get_bbox_coordinate(det_res):
    bbox_x1 = int(float(det_res["bbox"][0]))
    bbox_y1 = int(float(det_res["bbox"][1]))
    bbox_x2 = int(float(det_res["bbox"][2]))
    bbox_y2 = int(float(det_res["bbox"][3]))
    return bbox_x1, bbox_y1, bbox_x2, bbox_y2

def get_anchor_pts(pt, angle, width, left=False):
    angle = angle * np.pi / 180.
    if not left:
        anchor_x_length = width - 1 - pt[0].reshape(-1,1)
        # cos(anchor_angle) = anchor_x_len / anchor_len
        # anchor_len = anchor_x_len / cos(anchor_angle)
        anchor_length = anchor_x_length / np.cos(angle * np.pi / 180.)
        anchor_edge_y = pt[1].reshape(-1,1) + np.sin(angle) * anchor_length
        anchor_edge_x = (width - 1) * np.ones_like(anchor_edge_y)
        anchor_x_length = width // 2 - pt[0].reshape(-1,1)
        anchor_length = anchor_x_length / np.cos(angle * np.pi / 180.)
        anchor_middle_x = width // 2 * np.ones_like(anchor_edge_y) 
        anchor_middle_y = pt[1].reshape(-1,1) + np.sin(angle) * anchor_length
    else:
        anchor_x_length = pt[0].reshape(-1,1)
        # cos(anchor_angle) = anchor_x_len / anchor_len
        # anchor_len = anchor_x_len / cos(anchor_angle)
        anchor_length = anchor_x_length / np.cos(angle * np.pi / 180.)
        anchor_edge_y = pt[1].reshape(-1,1) + np.sin(angle) * anchor_length
        anchor_edge_x = np.zeros_like(anchor_edge_y)
        anchor_x_length = pt[0].reshape(-1,1) - width / 2
        anchor_length = anchor_x_length / np.cos(angle * np.pi / 180.)
        anchor_middle_x = width // 2 * np.ones_like(anchor_edge_y) 
        anchor_middle_y = pt[1].reshape(-1,1) + np.sin(angle) * anchor_length
    # return np.array((anchor_middle_x, anchor_middle_y, anchor_edge_x, anchor_edge_y))
    return np.stack([anchor_middle_x, anchor_middle_y, anchor_edge_x, anchor_edge_y], axis=-1)

def remove_non_alphanumeric(input_string):
    return ''.join([char for char in input_string if char.isalnum()])
    
def get_average_speed(speed_dict, side_r, time_buffer=0.1):
    ts_list = []
    for r in side_r:
        ts = float(r["filename"])
        ts_list.append(ts)
    ts_min = min(ts_list) - time_buffer
    ts_max = max(ts_list) + time_buffer
    speed_list = []
    for ts, s in speed_dict.items():
        if float(ts) > ts_min and float(ts) < ts_max:
            speed_list.append(s)
    print(f'num results: {len(side_r)}, num speeds: {len(speed_list)}')
    return sum(speed_list) / len(speed_list)

def get_nn_confidence_stat(side_r):
    nn_conf = []
    for r in side_r:
        nn_conf.append(float(r["score"]))
    nn_conf_max = max(nn_conf)
    nn_conf_avg = sum(nn_conf) / len(nn_conf)
    nn_conf_var = sum([((x - nn_conf_avg) ** 2) for x in nn_conf]) / len(nn_conf)

    return {
        "max": nn_conf_max,
        "mean": nn_conf_avg,
        "nn_conf_var": nn_conf_var,
    }

# read detection result
with open(detection_result_path, "r") as fd:
    det_res = json.load(fd)

# step 1: group frames based on timestamp
if RUN_STEP1:
    temp_cat = "s1_group"
    dst_path = mk_temp_dir_bc(temp_cat, rm_b4_mk=COPY_BY_GROUP, root=data_temp) #mk_temp_dir(temp_cat, data_label, rm_b4_mk=COPY_BY_GROUP, root=data_temp)
    group_idx = -1
    group_ts_last = -1.
    group_det_res = []
    group_det_res_curr = []
    det_res = sorted(det_res, key=lambda r: float(r["filename"]))
    for idx, r in enumerate(det_res):
        group_ts_curr = float(r["filename"])
        if group_ts_last < 0 or group_ts_curr - group_ts_last > group_max_time_diff:
            # start a new group
            group_det_res_curr_unique_count = len(set([r["filename"] for r in group_det_res_curr]))
            if group_det_res_curr_unique_count < group_min_frame_count:
                # frame number less than min num frames required
                if group_ts_last > 0:
                    # logging.debug(f'S1_GROUP: less than min requried frame count ({group_det_res_curr_unique_count}/{group_min_frame_count})')
                    print(f'*STEP1: less than min requried frame count ({group_det_res_curr_unique_count}/{group_min_frame_count})')
            else:
                group_idx += 1
                group_det_res.append(group_det_res_curr)
                # logging.info(f'S1_GROUP: group{group_idx} has {group_det_res_curr_unique_count} frames')
                print(f'*STEP1: group{group_idx} has {group_det_res_curr_unique_count} frames')
                if COPY_BY_GROUP:
                    copy_files(data_path, [r["filename"] for r in group_det_res_curr], dst_path, f'group{group_idx}')
            group_det_res_curr = []
        group_det_res_curr.append(r)
        group_ts_last = group_ts_curr
    # handle last group
    group_det_res_curr_unique_count = len(set([r["filename"] for r in group_det_res_curr]))
    if group_det_res_curr_unique_count < group_min_frame_count:
        # frame number less than min num frames required
        if group_ts_last > 0:
            print(f'*STEP1: less than min requried frame count ({group_det_res_curr_unique_count}/{group_min_frame_count})')
    else:
        group_idx += 1
        group_det_res.append(group_det_res_curr)
        print(f'*STEP1: group{group_idx} has {group_det_res_curr_unique_count} frames')
        if COPY_BY_GROUP:
            copy_files(data_path, [r["filename"] for r in group_det_res_curr], dst_path, f'group{group_idx}')

    with open(os.path.join(dst_path, "s1_group_det_res.json"), "w") as fd:
        json.dump(group_det_res, fd, indent=2)

# step 2: OCR 
if RUN_STEP2:
    from paddleocr import PaddleOCR, draw_ocr

    last_step_dst_path = os.path.join(data_temp, "s1_group", "s1_group_det_res.json")
    with open(last_step_dst_path, "r") as fd:
        group_det_res = json.load(fd)
        
    ocr = PaddleOCR(lang='en', show_log=False) 
    temp_cat = "s2_ocr"
    dst_path = mk_temp_dir_bc(temp_cat, rm_b4_mk=DRAW_OCR, root=data_temp)
    for group_idx, group_r in enumerate(group_det_res):
        print(f'*STEP2: working on group{group_idx}...')
        group_path = os.path.join(dst_path, f'group{group_idx}')
        if DRAW_OCR:
            os.mkdir(group_path)
        for frame_idx, r in enumerate(group_r):
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = get_bbox_coordinate(r)
            img_path = os.path.join(data_path, f'{r["filename"]}.png')
            image = Image.open(img_path).convert('RGB')
            roi = image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
            roi = roi.resize((400, 600))
            roi = np.array(roi)

            ocr_res = ocr.ocr(roi, cls=False)
            ocr_res = ocr_res[0]
            ocr_text = []
            print(f'***frame{frame_idx}***')
            if ocr_res is None:
                print("no OCR detection. skipping...")
            else:
                # get lower case ocr detected texts
                ocr_text = [text[1][0].lower() for text in ocr_res]
                print(ocr_text)
                # draw result
                if DRAW_OCR:
                    boxes = [line[0] for line in ocr_res]
                    txts = [line[1][0] for line in ocr_res]
                    scores = [line[1][1] for line in ocr_res]
                    im_show = draw_ocr(roi, boxes, txts, scores, font_path='/usr/share/fonts/truetype/lato/Lato-Regular.ttf')
                    im_show = Image.fromarray(im_show)
                    im_show.save(os.path.join(group_path, f'{r["filename"]}_{frame_idx}.png'))
            
            group_det_res[group_idx][frame_idx]["paddleocr"] = ocr_text

    with open(os.path.join(dst_path, "s2_group_det_res.json"), "w") as fd:
        json.dump(group_det_res, fd, indent=2)

# step 3: filter based on anchor (distance to anchor), bbox area change considering relative time difference to anchor source, and aspect ratio
if RUN_STEP3:
    from Levenshtein import distance

    filtered_group_r = {}

    last_step_dst_path = os.path.join(data_temp, "s2_ocr", "s2_group_det_res.json")
    with open(last_step_dst_path, "r") as fd:
        group_det_res = json.load(fd)
    
    temp_cat = "s3_anchor"
    dst_path = mk_temp_dir_bc(temp_cat, rm_b4_mk=VIS_ANCHOR, root=data_temp)
    for group_idx, group_r in enumerate(group_det_res):
        print(f'*STEP3: processing group {group_idx}...')
        anchor_src_count = 0
        anchor_src_bbox_ctr = []
        anchor_src_det_res = []
        anchor_src_frame_idx = []
        filtered_r = {
            "left": [],
            "left_anchor_src": [],
            "left_frame_idx": [],
            "left_anchor_src_frame_idx": [],
            "left_anchor_src_excluded_frame_idx": [],
            "left_excluded_frame_idx": [],
            "left_incomplete_last_frame": False,
            "right": [],
            "right_anchor_src": [],
            "right_frame_idx": [],
            "right_anchor_src_frame_idx": [],
            "right_anchor_src_excluded_frame_idx": [],
            "right_excluded_frame_idx": [],
            "right_incomplete_last_frame": False,
            "unassigned": [],
            "unassigned_frame_idx": [],
            "frame_num": 0,
        }
        for frame_idx ,frame_r in enumerate(group_r):
            # print(f'processing frame {frame_idx}...')
            ocr_text = frame_r["paddleocr"]
            min_speed_ldist = 5
            min_limit_ldist = 5
            for text in ocr_text:
                min_speed_ldist = min(min_speed_ldist, distance(text, "speed"))
                min_limit_ldist = min(min_limit_ldist, distance(text, "limit"))
                if min_speed_ldist <= max_speed_levenshtein_dist and min_limit_ldist <= max_limit_levenshtein_dist:
                    anchor_src_count += 1
                    # print(f'anchor candidate {ocr_text}')
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = get_bbox_coordinate(frame_r)
                    ctr_x, ctr_y = (bbox_x1 + bbox_x2) // 2, (bbox_y1 + bbox_y2) // 2
                    anchor_src_bbox_ctr.append([ctr_x, ctr_y])
                    anchor_src_det_res.append(frame_r)
                    anchor_src_frame_idx.append(frame_idx)
                    break
        print(f'anchor src count {anchor_src_count}')

        if anchor_src_count > 0:
            # split by left/right
            anchor_src_left_frame_idx, anchor_src_right_frame_idx = [], []
            anchor_src_left_bbox_ctr, anchor_src_right_bbox_ctr = [], []
            anchor_src_left_det_res, anchor_src_right_det_res = [], []
            for frame_idx, bbox_ctr, det_res in zip(anchor_src_frame_idx, anchor_src_bbox_ctr, anchor_src_det_res):
                if bbox_ctr[0] < width // 2:
                    anchor_src_left_frame_idx.append(frame_idx)
                    anchor_src_left_bbox_ctr.append(bbox_ctr)
                    anchor_src_left_det_res.append(det_res)
                else:
                    anchor_src_right_frame_idx.append(frame_idx)
                    anchor_src_right_bbox_ctr.append(bbox_ctr)
                    anchor_src_right_det_res.append(det_res)

            sides = []
            if len(anchor_src_left_bbox_ctr) > 0:
                sides.append("left")
            if len(anchor_src_right_bbox_ctr) > 0:
                sides.append("right")

            left_frame_idx, right_frame_idx = [], []
            left_det_res, right_det_res = [], []
            for frame_idx, frame_r in enumerate(group_r):
                side_x1, side_y1, side_x2, side_y2 = get_bbox_coordinate(frame_r)
                side_x_ctr, side_y_ctr = (side_x1 + side_x2)//2, (side_y1 + side_y2)//2
                if side_x_ctr < width//2:
                    left_det_res.append(frame_r)
                    left_frame_idx.append(frame_idx)
                else:
                    right_det_res.append(frame_r)
                    right_frame_idx.append(frame_idx)

            if VIS_ANCHOR:
                im = Image.open(os.path.join(data_path, f'{group_r[-2]["filename"]}.png'))
                draw = ImageDraw.Draw(im)

            for side in sides:
                if side == "left":
                    anchor_angles = np.arange(anchor_left_min_deg, anchor_left_max_deg, anchor_resolution_deg).reshape(1,-1)
                    anchors = get_anchor_pts(np.array(anchor_src_left_bbox_ctr, dtype=np.int32).T, anchor_angles, width, left=True)
                    anchor_src_side_frame_idx = anchor_src_left_frame_idx
                    anchor_src_side_bbox_ctr = anchor_src_left_bbox_ctr
                    anchor_src_side_det_res = anchor_src_left_det_res
                    side_frame_idx = left_frame_idx
                    side_det_res = left_det_res
                elif side == "right":
                    anchor_angles = np.arange(anchor_right_min_deg, anchor_right_max_deg, anchor_resolution_deg).reshape(1,-1)
                    anchors = get_anchor_pts(np.array(anchor_src_right_bbox_ctr, dtype=np.int32).T, anchor_angles, width)
                    anchor_src_side_frame_idx = anchor_src_right_frame_idx
                    anchor_src_side_bbox_ctr = anchor_src_right_bbox_ctr
                    anchor_src_side_det_res = anchor_src_right_det_res
                    side_frame_idx = right_frame_idx
                    side_det_res = right_det_res
                else:
                    raise NotImplementedError
                
                distance_to_anchor = np.zeros((len(side_frame_idx), *anchors.shape[:-1]))   # [num of frame per side * num of anchor src * num of anchor per anchor src]
                for sidx, sr in enumerate(side_det_res):
                    x1, y1, x2, y2 = get_bbox_coordinate(sr)
                    x_ctr, y_ctr = (x1+x2)//2, (y1+y2)//2
                    distance_to_anchor[sidx,:,:] = distance_pt2line(anchors[:,:,0],
                                                                    anchors[:,:,1],
                                                                    anchors[:,:,2],
                                                                    anchors[:,:,3],
                                                                    x_ctr,
                                                                    y_ctr)

                q_distance_to_anchor = distance_to_anchor < distance_threshold

                q_bbox_area = np.zeros_like(q_distance_to_anchor)
                # q_x = np.zeros_like(q_distance_to_anchor)
                for aidx in range(len(anchor_src_side_det_res)):
                    afidx = anchor_src_side_frame_idx[aidx]
                    ar = anchor_src_side_det_res[aidx]
                    ax1, ay1, ax2, ay2 = get_bbox_coordinate(ar)
                    ax_diff, ay_diff = ax2 - ax1, ay2 - ay1
                    abbox_area = ax_diff * ay_diff
                    bbox_area_before_anchor, bbox_area_after_anchor = [], []
                    # x_before_anchor, x_after_anchor = [], []
                    for sidx in range(len(side_frame_idx)):
                        sfidx = side_frame_idx[sidx]
                        sr = side_det_res[sidx]
                        sx1, sy1, sx2, sy2 = get_bbox_coordinate(sr)
                        sx_diff, sy_diff = sx2 - sx1, sy2 - sy1
                        sbbox_area = sx_diff * sy_diff
                        if sfidx < afidx:
                            bbox_area_before_anchor.append(sbbox_area)
                        elif sfidx > afidx:
                            if sidx < len(side_frame_idx) - 1:
                                bbox_area_after_anchor.append(sbbox_area)
                            else:
                                yx_ratio = sy_diff / sx_diff
                                if yx_ratio > 1.6:
                                    bbox_area_after_anchor.append(0.0)
                                    if side == "left":
                                        filtered_r["left_incomplete_last_frame"] = True
                                    elif side == "right":
                                        filtered_r["right_incomplete_last_frame"] = True

                                else:
                                    bbox_area_after_anchor.append(sbbox_area)
                        else:
                            anchor_side_idx = sidx
                    q_bbox_area_side_indices_before_anchor = longest_increasing_subsequence_indices_greater_than_constant(bbox_area_before_anchor, abbox_area, less_than_constant=True)
                    q_bbox_area_side_indices_before_anchor.append(anchor_side_idx)
                    q_bbox_area_side_indices_after_anchor = longest_increasing_subsequence_indices_greater_than_constant(bbox_area_after_anchor, abbox_area)
                    q_bbox_area_side_indices_after_anchor = [i+anchor_side_idx+1 for i in q_bbox_area_side_indices_after_anchor]
                    q_bbox_area_side_indices = q_bbox_area_side_indices_before_anchor + q_bbox_area_side_indices_after_anchor
                    # print(q_bbox_area_side_indices)
                    for sidx in q_bbox_area_side_indices:
                        q_bbox_area[sidx, aidx, :] = np.ones_like(q_bbox_area[0, 0, :])

                    if BYPASS_Q_BBOX_AREA:
                        q_bbox_area = np.ones_like(q_distance_to_anchor)
                
                qualification_mask = np.logical_and(q_distance_to_anchor, q_bbox_area)
                # qualification_mask = np.logical_and(q_x, qualification_mask)

                distance_qualified_count = np.sum(qualification_mask, axis=0, keepdims=True)
                distance_qualified_count_max = distance_qualified_count.max()

                distance_qualified_count_max_idxs = np.argwhere(distance_qualified_count == distance_qualified_count_max)
                sum_distance_qualified = np.sum(distance_to_anchor * qualification_mask, axis=0, keepdims=True)
                min_sum_distance_qualified_idx = np.argmin(sum_distance_qualified[
                    distance_qualified_count_max_idxs[:,0],
                    distance_qualified_count_max_idxs[:,1],
                    distance_qualified_count_max_idxs[:,2],
                ])
                best_anchor_src_side_id = distance_qualified_count_max_idxs[min_sum_distance_qualified_idx][1]
                best_anchor_anchor_id = distance_qualified_count_max_idxs[min_sum_distance_qualified_idx][2]
                print(f'best anchor {side} - anchor src frame id: {anchor_src_side_frame_idx[best_anchor_src_side_id]}; anchor id: {best_anchor_anchor_id}; num of frames: {distance_qualified_count_max}')

                # check if all anchor srcs are in the qualifed frame list
                # if not, more than one sign may be in the side
                qualified_sfidx = []
                excluded_sfidx = []
                for sidx, sfidx in enumerate(side_frame_idx):
                    if qualification_mask[sidx, best_anchor_src_side_id, best_anchor_anchor_id]:
                        qualified_sfidx.append(sfidx)
                    else:
                        excluded_sfidx.append(sfidx)
                
                side_anchor_src_excluded_frame_idx = []
                for afidx in anchor_src_side_frame_idx:
                    if afidx not in qualified_sfidx:
                        # side_anchor_src_excluded_num += 1
                        side_anchor_src_excluded_frame_idx.append(afidx)
                        # print(f'***STEP3-potentialy more than one sign on {side} side (frame#{afidx})')
                if side == "left":
                    filtered_r["left_anchor_src_excluded_frame_idx"].append(side_anchor_src_excluded_frame_idx)
                elif side == "right":
                    filtered_r["right_anchor_src_excluded_frame_idx"].append(side_anchor_src_excluded_frame_idx)
                
                # split det res by side and store in a dict
                if side == "left":
                    filtered_r["left_frame_idx"].append(qualified_sfidx)
                    filtered_r["left"].append([group_r[sfidx] for sfidx in qualified_sfidx])
                    filtered_r["left_anchor_src"].append(side_det_res[best_anchor_src_side_id])
                    filtered_r["left_anchor_src_frame_idx"].append(anchor_src_side_frame_idx[best_anchor_src_side_id])
                    filtered_r["left_excluded_frame_idx"].append(excluded_sfidx)
                elif side == "right":
                    filtered_r["right_frame_idx"].append(qualified_sfidx)
                    filtered_r["right"].append([group_r[sfidx] for sfidx in qualified_sfidx])
                    filtered_r["right_anchor_src"].append(side_det_res[best_anchor_src_side_id])
                    filtered_r["right_anchor_src_frame_idx"].append(anchor_src_side_frame_idx[best_anchor_src_side_id])
                    filtered_r["right_excluded_frame_idx"].append(excluded_sfidx)

                if VIS_ANCHOR:
                    for sidx, sr in enumerate(side_det_res):
                        x1, y1, x2, y2 = get_bbox_coordinate(sr)
                        x_ctr = (x1+x2)//2
                        y_ctr = (y1+y2)//2
                        color = "green" if side=="right" else "blue"
                        draw.rectangle((x1, y1, x2, y2), outline=color if qualification_mask[sidx,best_anchor_src_side_id,best_anchor_anchor_id] else "red", width=5)
                        # draw.rectangle((x_ctr-5, y_ctr-5, x_ctr+5, y_ctr+5), fill=color if qualification_mask[si,best_anchor_frame_id,best_anchor_anchor_id] else "red", width=3)
                        font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Regular.ttf", 50)
                        draw.text((x_ctr,y_ctr), str(sidx), fill="white", font=font) 
                    draw.line(list(anchors[best_anchor_src_side_id, best_anchor_anchor_id, :]), fill="yellow", width=10)
            
            assigned_frame_idx = []
            for fidx in filtered_r["left_frame_idx"]:
                assigned_frame_idx += fidx
            for fidx in filtered_r["right_frame_idx"]:
                assigned_frame_idx += fidx
            filtered_r["frame_num"] = len(group_r)
            filtered_r["unassigned_frame_idx"] = [fidx for fidx in range(filtered_r["frame_num"]) if fidx not in assigned_frame_idx] 
            filtered_r["unassigned"] = [group_det_res[fidx] for fidx in filtered_r["unassigned_frame_idx"]]
            filtered_group_r[group_idx] = filtered_r

            if VIS_ANCHOR:
                im.save(os.path.join(dst_path, f'group{group_idx}.png'))
    
    with open(os.path.join(dst_path, "s3_group_det_res.json"), "w") as fd:
        json.dump(filtered_group_r, fd, indent=2)

# step 4: assign score
#   - longitudinal visibility
#   - lateral visibility
#   - content completeness
if RUN_STEP4:
    last_step_dst_path = os.path.join(data_temp, "s3_anchor", "s3_group_det_res.json")
    with open(last_step_dst_path, "r") as fd:
        group_det_res = json.load(fd)

    SPEED_FILE_NA = len(speed_file_path) == 0
    if not SPEED_FILE_NA:
        with open(speed_file_path, "r") as fd:
            speed_dict = json.load(fd)

    group_stat = {}
        
    non_empty_group_idx = group_det_res.keys()
    for group_idx in group_det_res.keys():
        group_r = group_det_res[group_idx]
        group_r_left = group_r["left"]
        group_r_right = group_r["right"]
        group_r_left_excluded_fidx = group_r["left_excluded_frame_idx"]
        group_r_right_excluded_fidx = group_r["right_excluded_frame_idx"]
        group_r_unassigned = group_r["unassigned"]
        group_r_left_excluded_anchor_src = group_r["left_anchor_src_excluded_frame_idx"]
        group_r_right_excluded_anchor_src = group_r["right_anchor_src_excluded_frame_idx"]
        # describe the result
        print(f'*STEP4: Group{group_idx} has {group_r["frame_num"]} frame. Left side has {len(group_r_left)} signes with {[len(r) for r in group_r_left] if len(group_r_left)>0 else "NA"} frames each sign. Right side has {len(group_r_right)} signs with {[len(r) for r in group_r_right] if len(group_r_right)>0 else "NA"} frames each sign. {len(group_r_unassigned)} frames are unassigned.')

        left_stat = {
            "b_sign": False,
            "num_frames": len(group_r_left[0]) if group_r_left else 0,
            "num_frames_digits": 0,
            "num_frames_excluded": 0 if len(group_r_left_excluded_fidx)==0 else len(group_r_left_excluded_fidx[0]),
            "max_num_frames_same_digits": 0,
            "speed_limit_prediction": None,
            "all_speed_limit_detection": {},
            "num_unassigned_anchor_source": 0,
            "average_vehicle_speed": None,
            "min_num_pixels_bbox": 0,
            "max_num_pixels_bbox": 0,
            "nn_confidence_stat": None,
            "incomplete_last_frame": group_r["left_incomplete_last_frame"],
            "num_legible_frames": 0,
            "legibility_time_fn": 0.0,
        }
        right_stat = {
            "b_sign": False,
            "num_frames": len(group_r_right[0]) if group_r_right else 0,
            "num_frames_digits": 0,
            "num_frames_excluded": 0 if len(group_r_right_excluded_fidx)==0 else len(group_r_right_excluded_fidx[0]),
            "max_num_frames_same_digits": 0,
            "speed_limit_prediction": None,
            "all_speed_limit_detection": {},
            "num_unassigned_anchor_source": 0,
            "average_vehicle_speed": None,
            "min_num_pixels_bbox": 0,
            "max_num_pixels_bbox": 0,
            "nn_confidence_stat": None,
            "incomplete_last_frame": group_r["right_incomplete_last_frame"],
            "num_legible_frames": 0,
            "legibility_time_fn": 0.0,
        }

        # potential of multiple signs on the same side that are too close to each other
        if len(group_r_left_excluded_anchor_src) > 0 and len(group_r_left_excluded_anchor_src[0]):
            print("LEFT side potentially has more than 1 sign")
            left_stat["num_unassigned_anchor_source"] = len(group_r_left_excluded_anchor_src[0])
        if len(group_r_right_excluded_anchor_src) > 0 and len(group_r_right_excluded_anchor_src[0]):
            print("RIGHT side potentially has more than 1 sign")
            right_stat["num_unassigned_anchor_source"] = len(group_r_right_excluded_anchor_src[0])
        # detectability of speed limit value
        def get_speed_count_pairs(group_texts):
            num_frames_w_digit = 0
            speed_limit_values = []
            b_first_frame_w_digit = True
            first_frame_w_digit_sidx = -1
            last_frame_w_digit_sidx = -1
            for idx, texts in enumerate(group_texts):
                b_frame_w_digit = False
                for t in texts:
                    if t.isdigit():
                        speed_limit_values.append(t)
                        b_frame_w_digit = True
                        if b_first_frame_w_digit:
                            first_frame_w_digit_sidx = idx
                            b_first_frame_w_digit = False
                if b_frame_w_digit:
                    num_frames_w_digit += 1
                    last_frame_w_digit_sidx = idx
            speed_limit_value_set = set(speed_limit_values)
            speed_limit_value_count_pair = {v:0 for v in speed_limit_value_set}
            for v in speed_limit_values:
                speed_limit_value_count_pair[v] += 1 
            return speed_limit_value_count_pair, num_frames_w_digit, first_frame_w_digit_sidx, last_frame_w_digit_sidx
        if len(group_r_left) > 0:
            left_stat["b_sign"] = True
            group_r_left_0_text = [r["paddleocr"] for r in group_r_left[0]]
            speed_limit_value_count_pair, num_frames_w_digit, first_frame_w_digit_sidx, last_frame_w_digit_sidx = get_speed_count_pairs(group_r_left_0_text)
            # calculate legibility time using filename (timestamp)
            legibility_time_fn = 0.0
            if first_frame_w_digit_sidx >= 0:
                if first_frame_w_digit_sidx == last_frame_w_digit_sidx:
                    legibility_time_fn = 0.2
                else:
                    first_frame_fn = float(group_r_left[0][first_frame_w_digit_sidx]["filename"])
                    last_frame_fn = float(group_r_left[0][last_frame_w_digit_sidx]["filename"])
                    legibility_time_fn = last_frame_fn - first_frame_fn
            left_stat["legibility_time_fn"] = legibility_time_fn
            left_stat["num_legible_frames"] = left_stat["num_frames"] - first_frame_w_digit_sidx if first_frame_w_digit_sidx >= 0 else 0
            # print(f'LEFT: SL values {speed_limit_value_count_pair}')
            print("***LEFT***")
            print(f'num frames w/ digits: {num_frames_w_digit}/{len(group_r_left_0_text)}.')
            left_stat["num_frames_digits"] = num_frames_w_digit
            left_stat["average_vehicle_speed"] = -1 if SPEED_FILE_NA else get_average_speed(speed_dict, group_r_left[0])
            left_stat["nn_confidence_stat"] = get_nn_confidence_stat(group_r_left[0])
            if num_frames_w_digit > 0:
                speed_limit_prediction = max(speed_limit_value_count_pair, key=speed_limit_value_count_pair.get)
                print(f'Speed limit: {speed_limit_prediction} w/ confidence {speed_limit_value_count_pair[speed_limit_prediction]/num_frames_w_digit:.4f}')
                left_stat["speed_limit_prediction"] = speed_limit_prediction
                if SPEED_FILE_NA:
                    left_stat["average_vehicle_speed"] = int(speed_limit_prediction) * 0.44704
                left_stat["all_speed_limit_detection"] = speed_limit_value_count_pair
                left_stat["max_num_frames_same_digits"] = speed_limit_value_count_pair[speed_limit_prediction]
                # number of pixel 
                x1, y1, x2, y2 = get_bbox_coordinate(group_r_left[0][0])
                left_stat["min_num_pixels_bbox"] = (x2-x1)*(y2-y1)
                x1, y1, x2, y2 = get_bbox_coordinate(group_r_left[0][-2] if len(group_r_left[0])>1 else group_r_left[0][-1])
                left_stat["max_num_pixels_bbox"] = (x2-x1)*(y2-y1)

            else:
                print("Not able to predict speed limit.")
            print("******")
        if len(group_r_right) > 0:
            right_stat["b_sign"] = True
            group_r_right_0_text = [r["paddleocr"] for r in group_r_right[0]]
            speed_limit_value_count_pair, num_frames_w_digit, first_frame_w_digit_sidx, last_frame_w_digit_sidx = get_speed_count_pairs(group_r_right_0_text)

            # calculate legibility time using filename (timestamp)
            legibility_time_fn = 0.0
            if first_frame_w_digit_sidx >= 0:
                if first_frame_w_digit_sidx == last_frame_w_digit_sidx:
                    legibility_time_fn = 0.2
                else:
                    first_frame_fn = float(group_r_right[0][first_frame_w_digit_sidx]["filename"])
                    last_frame_fn = float(group_r_right[0][last_frame_w_digit_sidx]["filename"])
                    legibility_time_fn = last_frame_fn - first_frame_fn
            right_stat["legibility_time_fn"] = legibility_time_fn
            right_stat["num_legible_frames"] = right_stat["num_frames"] - first_frame_w_digit_sidx if first_frame_w_digit_sidx >= 0 else 0
            print("***RIGHT***")
            print(f'num frames w/ digits: {num_frames_w_digit}/{len(group_r_right_0_text)}.')
            right_stat["num_frames_digits"] = num_frames_w_digit
            right_stat["average_vehicle_speed"] = -1 if SPEED_FILE_NA else get_average_speed(speed_dict, group_r_right[0])
            right_stat["nn_confidence_stat"] = get_nn_confidence_stat(group_r_right[0])
            if num_frames_w_digit > 0:
                speed_limit_prediction = max(speed_limit_value_count_pair, key=speed_limit_value_count_pair.get)
                print(f'Speed limit: {speed_limit_prediction} w/ confidence {speed_limit_value_count_pair[speed_limit_prediction]/num_frames_w_digit:.4f}')
                right_stat["speed_limit_prediction"] = speed_limit_prediction
                if SPEED_FILE_NA:
                    right_stat["average_vehicle_speed"] = int(speed_limit_prediction) * 0.44704
                right_stat["all_speed_limit_detection"] = speed_limit_value_count_pair
                right_stat["max_num_frames_same_digits"] = speed_limit_value_count_pair[speed_limit_prediction]
                # number of pixel 
                x1, y1, x2, y2 = get_bbox_coordinate(group_r_right[0][0])
                right_stat["min_num_pixels_bbox"] = (x2-x1)*(y2-y1)
                x1, y1, x2, y2 = get_bbox_coordinate(group_r_right[0][-2] if len(group_r_right)>1 else group_r_right[0][-1])
                right_stat["max_num_pixels_bbox"] = (x2-x1)*(y2-y1)
            else:
                print("Not able to predict speed limit.")
            print("******")

        group_stat[group_idx] = {
            "left": left_stat,
            "right": right_stat,
        }

    temp_cat = "s4_stat"
    dst_path = mk_temp_dir_bc(temp_cat, root=data_temp)
    with open(os.path.join(dst_path, "s4_group_det_res.json"), "w") as fd:
        json.dump(group_stat, fd, indent=2)

if RUN_STEP5:
    last_step_dst_path = os.path.join(data_temp, "s4_stat", "s4_group_det_res.json")
    with open(last_step_dst_path, "r") as fd:
        group_det_res = json.load(fd)

    sign_criterion = {}
    for key in group_det_res.keys():
        group_r = group_det_res[key]
        print(f'*Step5-Processing Group{key}...')
        sides = ["left", "right"]
        side_c = {
            "left": {},
            "right": {},
        }
        for side in sides:
            c = {
                "reaction_time": None,
                "reaction_distance": None,
                "legibility_distance": None,
                "legibility_time": None,
                "conspicuity": None,
                "glance_legibility": None,
                "understandability": None,
                "overall_score": None,
                "legibility_time_score": None,
                "conspicuity_score": None,
                "glance_legibility_score": None,
                "understandability_score": None
            }
            if side == "left":
                side_r = group_r["left"]
            elif side == "right":
                side_r = group_r["right"]
            else:
                raise NotImplementedError
            if side_r["b_sign"]:
                # 1. reaction distance(time) -
                #  num of frames * (1/frequency) -
                # c["reaction_distance"] = side_r["num_frames"] / camera_capture_freq * side_r["average_vehicle_speed"]
                # c["reaction_time"] = c["reaction_distance"] / (float(side_r["speed_limit_prediction"]) * 0.44704) if side_r["speed_limit_prediction"] else None
                # if side_r["speed_limit_prediction"]:
                #     print(f'1.reaction distance {c["reaction_distance"]}m; reaction time {side.upper()}: {c["reaction_time"]:.2f} seconds ({side_r["num_frames"] / camera_capture_freq} seconds @ {side_r["average_vehicle_speed"]:.2f} m/s for {side_r["speed_limit_prediction"]} MPH)')
                # else:
                #     print(f'1.reaction distance {c["reaction_distance"]}m; reaction time {side.upper()}: NA')
                # 2. legibility distance -
                # c["legibility_distance"] = (side_r["num_legible_frames"] - side_r["incomplete_last_frame"]) / camera_capture_freq * side_r["average_vehicle_speed"]
                # c["legibility_time"] = c["legibility_distance"] / (float(side_r["speed_limit_prediction"]) * 0.44704) if side_r["speed_limit_prediction"] else None
                # print(f'2.legibility distance {side.upper()}: {c["legibility_distance"]:.2f} meters ({(side_r["num_legible_frames"] - side_r["incomplete_last_frame"]) / camera_capture_freq} seconds @ {side_r["average_vehicle_speed"]:.2f} m/s)')
                # print(f'legibility time: {c["legibility_time"] if side_r["speed_limit_prediction"] else "NA"}')
                c["legibility_time"] = side_r["legibility_time_fn"]
                # print(f'legibility time: {c["legibility_time"]}')
                # 3. conspicuity -
                #  num of excluded frames (FPs) / num of frames
                c["conspicuity"] = {
                    "num_false_positive": side_r["num_frames_excluded"],
                    "num_true_positive": side_r["num_frames"],
                    "ratio": side_r["num_frames"]/(side_r["num_frames"] + side_r["num_frames_excluded"])
                }
                # print(f'3.conspicuity {side.upper()}: {c["conspicuity"]["num_false_positive"]}/{c["conspicuity"]["num_true_positive"]} - {c["conspicuity"]["ratio"]}')
                # 4. glance legibility -
                #  num of frames the same of speed limit prediction / num of frames within a certain distance -
                #  TODO: decide the certain distance within
                c["glance_legibility"] = side_r["max_num_frames_same_digits"] / (side_r["num_legible_frames"] - side_r["incomplete_last_frame"]) if (side_r["num_legible_frames"] - side_r["incomplete_last_frame"]) else 0
                # print(f'4.glance legibility {side.upper()}: {side_r["max_num_frames_same_digits"]}/{side_r["num_legible_frames"] - side_r["incomplete_last_frame"]}')
                # 5. understandability
                c["understandability"] = side_r["nn_confidence_stat"]
                # print(f'5.understandability {side.upper()}: {side_r["nn_confidence_stat"]}')
                # overall score calculation
                lmap_u = lmap(c["understandability"]["max"], lmap_u_max, lmap_u_min)
                c["understandability_score"] = lmap_u
                print(f'1.understandability score: {lmap_u:.2f}')
                lmap_l = lmap(c["legibility_time"], lmap_l_max, lmap_l_min) if c["legibility_time"] else 0.0
                c["legibility_time_score"] = lmap_l
                print(f'2.legibility time score: {lmap_l:.2f}')
                lmap_c = lmap(c["conspicuity"]["ratio"], lmap_c_max, lmap_c_min)
                c["conspicuity_score"] = lmap_c
                print(f'3.conspicuity score: {lmap_c:.2f}')
                lmap_g = lmap(c["glance_legibility"], lmap_g_max, lmap_g_min)
                c["glance_legibility_score"] = lmap_g
                print(f'4.glance legibility score: {lmap_g:.2f}')
                c["overall_score"] = weight_u*lmap_u + weight_l*lmap_l + weight_c*lmap_c + weight_g*lmap_g
                print(f'OVERALL score: {c["overall_score"]:.2f}')

            else:
                print(f'No sign at {side.upper()}')
            print("------")

            if side == "left":
                side_c["left"] = c
            elif side == "right":
                side_c["right"] = c
        sign_criterion[key] = side_c

    
    temp_cat = "s5_criterion"
    dst_path = mk_temp_dir_bc(temp_cat, root=data_temp)
    with open(os.path.join(dst_path, "s5_group_det_res.json"), "w") as fd:
        json.dump(sign_criterion, fd, indent=2)

if RUN_STEP6:
    import matplotlib.pyplot as plt

    last_step_dst_path = os.path.join(data_temp, "s5_criterion", "s5_group_det_res.json")
    with open(last_step_dst_path, "r") as fd:
        group_det_res = json.load(fd)

    temp_cat = "s6_plot"
    dst_path = mk_temp_dir_bc(temp_cat, root=data_temp)

    sign_idxs = list(group_det_res.keys())
    # print(sign_idxs)
    sides = ["left", "right"]

    sign_idxs_side = []
    for i in sign_idxs:
        if group_det_res[i]["right"]["conspicuity"]:
            sign_idxs_side.append(i)
        if group_det_res[i]["left"]["conspicuity"]:
            sign_idxs_side.append(i + "_l")
    
    legibility_times = [group_det_res[i[:-2] if "_l" in i else i]["left" if "_l" in i else "right"]["legibility_time"] if group_det_res[i[:-2] if "_l" in i else i]["left" if "_l" in i else "right"]["legibility_time"] else 0.0 for i in sign_idxs_side]
    conspicuity_ratio = [group_det_res[i[:-2] if "_l" in i else i]["left" if "_l" in i else "right"]["conspicuity"]["ratio"] for i in sign_idxs_side]
    glance_legibility = [min(group_det_res[i[:-2] if "_l" in i else i]["left" if "_l" in i else "right"]["glance_legibility"], 1.0) for i in sign_idxs_side]
    understandability = [group_det_res[i[:-2] if "_l" in i else i]["left" if "_l" in i else "right"]["understandability"]["max"] for i in sign_idxs_side]
    
    fig, ax1 = plt.subplots()
    ln1 = ax1.scatter(sign_idxs_side, glance_legibility, label="glance legibility")
    ln2 = ax1.scatter(sign_idxs_side, understandability, label="understandability")
    ln3 = ax1.scatter(sign_idxs_side, conspicuity_ratio, label="conspicuity")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Percentage [%]")
    ax1.set_xlabel("Sign ID")
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ln4 = ax2.scatter(sign_idxs_side, legibility_times, label="legibility time", color="tab:red")
    ax2.set_ylim(-0.05, 3.05)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylabel("Time [s]")

    lns = [ln1, ln2, ln3, ln4]
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, bbox_to_anchor=(0.5, -0.2), loc='upper center', ncols=2)
    plt.title("Sign Evaluation")

    plt.tight_layout()

    plt.savefig(os.path.join(dst_path, f'sign_evaluation.png'))

# ****************************** Extract bbox from coco file ******************************
file_s4 = read_json(os.path.join(s4_dir, s4))
file_s1 = read_json(os.path.join(s1_dir, s1))
file_s5 = read_json(os.path.join(s5_dir, s5))
keys = list(file_s4)
sign_index, sign_side, spl_pred, sign_timestamp, sign_bbox, sign_score, sign_txt, sign_label, ocr_flag = [], [], [], [], [], [], [], [], []
leg_time, conspi_ratio, glan_leg, und_max, overall_score = [], [], [], [], []
score_leg, score_csp, score_gla, score_und = [], [], [] ,[]
co_x, co_y,co_w,co_l,co_id, ocr_prob = [],[],[],[],[],[]
sign_dict = {}
n = 0
rs = 1.0
for i in range(len(keys)):
    for side in ['left', 'right']:
        # print( '------- Index: ', keys[i], 'Side: ', side, '-------')
        sub_filename = []
        if file_s4[keys[i]][side]['b_sign']:
            n+=1
            print('Found sign on ', side, '--------------at ', file_s1[int(keys[i])][-1]['filename'])
            co_id.append(n)
            sign_index.append(keys[i])
            sign_side.append(side)
            spl_pred.append(file_s4[keys[i]][side]['speed_limit_prediction'])
            sign_timestamp.append(file_s1[int(keys[i])][-1]['filename']) # Pick the last timestamp for sign locating
            for j in range(len(file_s1[int(keys[i])])):
                sub_filename.append(file_s1[int(keys[i])][j]['filename'])
            sign_dict.update({file_s1[int(keys[i])][-1]['filename']: sub_filename})
            sign_label.append(file_s1[int(keys[i])][-1]['label'])
            if file_s1[int(keys[i])][-1]['filename'] == file_s1[int(keys[i])][-2]['filename']:
                # print(file_s1[int(keys[i])][-1]['bbox'][0])
                # print(img_size[0])
                if side == 'left':
                    if float(file_s1[int(keys[i])][-1]['bbox'][0]) < int(img_size[0]/2):
                        bbox = file_s1[int(keys[i])][-1]['bbox']
                        # print('On the left and use first')
                    else:
                        bbox = file_s1[int(keys[i])][-2]['bbox']
                        # print('On the left and use second')
                elif side == 'right':
                    if float(file_s1[int(keys[i])][-1]['bbox'][0]) >= int(img_size[0]/2):
                        bbox = file_s1[int(keys[i])][-1]['bbox']
                        # print('On the right and use first')
                    else:
                        bbox = file_s1[int(keys[i])][-2]['bbox']
                        # print('On the right and use second')
            else:
                bbox = file_s1[int(keys[i])][-1]['bbox']
            print('Keys: ', int(keys[i]), '--- bbox: ', bbox)
            co_x.append(float(bbox[0]))
            co_y.append(float(bbox[1])/rs)
            co_w.append(float(bbox[2]) - float(bbox[0])) #float(item['bbox'][2] is now x for lower right
            co_l.append(float(bbox[3])/rs - float(bbox[1])/rs) #float(item['bbox'][3] is now y for lower right
            sign_score.append(file_s1[int(keys[i])][-1]['score'])
            ocr = file_s1[int(keys[i])][-1]['ocr_output']
            txt = ''
            for j in range(len(ocr)):
                txt = txt + ocr[j][0] + ' '
            sign_txt.append(txt)

            if file_s5[keys[i]][side]['legibility_time'] is not None:    
                print('legibility_time: ', file_s5[keys[i]][side]['legibility_time'])
                leg_time.append( file_s5[keys[i]][side]['legibility_time'])
            else:
                leg_time.append(None)
            if file_s5[keys[i]][side]['legibility_time_score'] is not None:    # legibility_time_score
                print('legibility_time_score: ', file_s5[keys[i]][side]['legibility_time_score'])
                score_leg.append( file_s5[keys[i]][side]['legibility_time_score'])
            else:
                score_leg.append(None)
            if file_s5[keys[i]][side]['conspicuity'] is not None:         
                print('conspicuity ratio: ', file_s5[keys[i]][side]['conspicuity']['ratio'])
                conspi_ratio.append( file_s5[keys[i]][side]['conspicuity']['ratio'])
            else:
                conspi_ratio.append(None)
            if file_s5[keys[i]][side]['conspicuity_score'] is not None:         # conspicuity_score
                print('conspicuity_score: ', file_s5[keys[i]][side]['conspicuity_score'])
                score_csp.append( file_s5[keys[i]][side]['conspicuity_score'])
            else:
                score_csp.append(None)
            if file_s5[keys[i]][side]['glance_legibility'] is not None:         
                print('glance_legibility: ', file_s5[keys[i]][side]['glance_legibility'])
                glan_leg.append(file_s5[keys[i]][side]['glance_legibility'])
            else:
                glan_leg.append(None)
            if file_s5[keys[i]][side]['glance_legibility_score'] is not None:       # glance_legibility_score  
                print('glance_legibility_score: ', file_s5[keys[i]][side]['glance_legibility_score'])
                score_gla.append(file_s5[keys[i]][side]['glance_legibility_score'])
            else:
                score_gla.append(None)
            if file_s5[keys[i]][side]['understandability'] is not None:        
                print('understandability max: ', file_s5[keys[i]][side]['understandability']['max'])
                und_max.append( file_s5[keys[i]][side]['understandability']['max'])      
            else:
                und_max.append(None)
            if file_s5[keys[i]][side]['understandability_score'] is not None:         # understandability_score
                print('understandability_score: ', file_s5[keys[i]][side]['understandability_score'])
                score_und.append( file_s5[keys[i]][side]['understandability_score'])      
            else:
                score_und.append(None)
            if file_s5[keys[i]][side]['overall_score'] is not None:         
                print('overall_score: ', file_s5[keys[i]][side]['overall_score'])
                overall_score.append( file_s5[keys[i]][side]['overall_score'])      
            else:
                overall_score.append(None)
        else:
            pass
if len(sign_timestamp) > 0 and len(co_id) > 0:
    # Sort the timestamps and their id for the clustering 
    sign_timestamp_s, co_id_s= (list(t) for t in zip(*sorted(zip(sign_timestamp, co_id))))
    print('Extracting the sign recognition result - Complete')
    # ****************************** Main ******************************
    print('***Starting the sign Locating Process***')
    selected_lat, selected_lon, selected_spd,selected_depth, selected_depth_validity, selected_roi_u, selected_roi_v, selected_roi_x = [], [], [], [], [],[], [], []
    sign_id, sign_time, bbox = [],[],[]
    print(len(co_id), len(sign_score), len(ocr_flag))
    for i in range(len(co_id)): # New method that is applicable to the custom_coco
        
        # try:
        imgtime = sign_timestamp[i]
        gps_index = find_timestamp(latlon_t, imgtime)

        sign_depth = nominal_sign_depth
        sign_roi_x = nominal_sign_lateral
        if sign_side[i] == 'left':
            sign_roi_x = -1 * sign_roi_x 
        if unit_metadata['depth_map']:
            depthfile = os.path.join(depdir, imgtime+'.bin') # find depth file with the same image name (timestamp)
            depth_raw = np.fromfile(depthfile, dtype=np.uint8) #Construct an array from data in a text or binary file.
            start_point = (int(co_x[i]+ ra*co_w[i]),int(co_y[i]+ra*co_l[i])) #(int(x[i]),int(y[i]))
            end_point = (int(co_x[i]+(1-ra)*co_w[i]),int(co_y[i]+(1-ra)*co_l[i])) #(int(x[i]+w[i]),int(y[i]+l[i]))
            depth_val, valid, depth_values = depth_from_bbox(depth_raw, img_size, start_point, end_point)

            if valid: # True if len(depth_values) >= 4:
                if depth_val >= VALID_DEPTH_RANGE[0] and depth_val <= VALID_DEPTH_RANGE[1]: #if depth within valid range
                    print('Sign_id: ',co_id[i], 'Sign_info: ', sign_txt[i],' with mean depth value of ', "%.2f" % depth_val, '(max)', "%.2f" % max(depth_values), 
                            '(min)', "%.2f" % min(depth_values),' at timestamp ', "%.4f"%float(imgtime))
                    # selected_spd.append(spd[gps_index])
                    u, v = 1/2*(start_point[0]+end_point[0]), 1/2*(start_point[1]+end_point[1])
                    sign_roi_x = (u-cx)*depth_val/fx
                    selected_roi_u.append(u) #
                    selected_roi_v.append(v)
                    bbox.append((start_point, end_point))
                    sign_depth = depth_val + CAM_GPS_LONG
                else:
                    print('Sign at ', imgtime,' has depth out of selected range! Using nominal value...')
            else:
                print('***** Depth not valid at ', imgtime, '. Using nominal value... *****')
        else:
            print('Depth not available')     
        sign_id.append(co_id[i])
        sign_time.append(sign_timestamp[i])
        selected_depth.append(sign_depth) # distance between camera and target
        selected_roi_x.append(sign_roi_x) # fx,fy,cx,cy are Camera intrinsics
        selected_lat.append(lat[gps_index])
        selected_lon.append(lon[gps_index])
                        
        # except ValueError:
        #     pass

        # ****************************** Get veh/ sign poistion ******************************
    print('---------Detected sign times--------', sign_time)
    # East, North for select points
    sel_e1, sel_n1 = latlon2en(pp, selected_lat, selected_lon) # sign detected by ZED camera (PP is pyproj)
    e1, n1 = latlon2en(pp, lat, lon) # veh poistion
    d1_zed = np.array(selected_depth) #camera's x2_local
    x1_zed = np.array(selected_roi_x) #camera's y2_local
    # Sign east, north from zed camera
    e1_zed, n1_zed = [],[]
    displacement, dir = 0, 0
    for i in range(len(sel_e1)):
        if imu_switch:
            imu_index = find_timestamp(latlon_t, sign_time[i])
            hdg1 = np.pi/180*(90-heading_deg[imu_index]) # convert azimuth to heading
            # hdg1 = heading_deg[imu_index] # te data in file is already heading, not azimuth
            print('Heading angle--- from imu (selected): ', hdg1, 'from calculation: ', find_hdg(e1,n1,sel_e1[i]))
        else:
            imu_index = find_timestamp(latlon_t, sign_time[i])
            hdg1 = find_hdg(e1,n1,sel_e1[i])
            print('Heading angle--- from imu: ', np.pi/180*(90-heading_deg[imu_index]), 'from calculation (selected): ', hdg1)
        
        e_zed, n_zed = local2global(sel_e1[i], sel_n1[i], hdg1, d1_zed[i], -x1_zed[i])
        e1_zed.append(e_zed)
        n1_zed.append(n_zed)
    lat_zed, lon_zed = en2latlon(pp,n1_zed,e1_zed)
    sign_geojson(lon_zed, lat_zed, sign_time, selected_depth, selected_roi_x, sign_side, sign_dict, sign_score, leg_time, conspi_ratio, glan_leg, und_max, score_leg, score_csp, score_gla, score_und, overall_score, sign_txt, rootdir, geojson_name)

    print('***Sign locating Process Completed***')
else:
    print('No sign detected in ', unitdir)
