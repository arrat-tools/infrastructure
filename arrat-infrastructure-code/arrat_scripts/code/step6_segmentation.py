#!/usr/bin/env python
import argparse, json, os, json, argparse, glob, pyproj
import numpy as np
np.float = float

# python frames_to_segments.py --settings /path/to/inputs.json

inputs_filename = 'inputs.json'
unitname = 'unit'
frames_filename = 'frames.json'
session_filename = "session.geojson"
mile2meter = 1609.34


def segment_indices_from_frames(frames_list, ds):
    unit_segment_indices, ended_indices = [], False
    unit_end_lrs = frames_list[-1]['dist']
    last_index = len(frames_list)-1
    for i, frame_dict in enumerate(frames_list):
        # make segments
        if i == 0:
            prev_end_lrs = 	frame_dict['dist']
            seg_start_index = i
        # check if it is time to end
        if not ended_indices:
            if abs(unit_end_lrs-frame_dict['dist']) <= 0.5*ds:
                unit_segment_indices.append([seg_start_index, last_index])
                prev_end_lrs = frame_dict['dist']
                ended_indices = True
        if not ended_indices:
            if abs(frame_dict['dist']-prev_end_lrs) >= ds:
                unit_segment_indices.append([seg_start_index, i])
                prev_end_lrs = frame_dict['dist']
                seg_start_index = i
    return unit_segment_indices


def get_seg_dist(frame_list):
    for i, frame_dict in enumerate(frame_list):
        if i == 0:
            bad_dist = 0.0
            total_dist = 0.0
            bad_count = 0
            total_count = 0
            start_lrs = frame_dict['lrs']
        total_dist = total_dist + frame_dict['path_increment']
        total_count += 1
    end_lrs = frame_dict['lrs']
    total_lrs = abs(end_lrs - start_lrs)
    return total_count, total_dist

parser = argparse.ArgumentParser(
    description="This script accepts one input filepath and unit number to process ego lane line properties."
)
parser.add_argument(
     "--datadir", default="", help="Session data directory"
)

args = parser.parse_args()

# Inputs
datadir = args.datadir

aggregate_geofile = os.path.join(datadir, session_filename)

# how many units exist in session
unitlist = glob.glob(os.path.join(datadir, unitname+'*'))
unitlist.sort()

# init geojson
geojson_dict = {
	"type": "FeatureCollection",
    "summary" : {},
	"features": [
	]
}
outputs_json = {}
segment_outputs_list = []
# init aggregated output

total_frames, total_lrs, total_dist = 0, 0.0, 0.0
seg_num = 0
for unitdir in unitlist:
    print("\n****     Processing unit", unitdir, "    ****")

    inputs_file = os.path.join(unitdir, inputs_filename)
    f = open(inputs_file)
    inputs = json.load(f)
    f.close()
    segment_length = inputs['segment_length'] # miles

    framesfile = os.path.join(unitdir, frames_filename)
    f = open(framesfile)
    frames_list = json.load(f)['laneline_frames']
    f.close()	
    unit_segment_indices = segment_indices_from_frames(frames_list, segment_length*mile2meter)		

    unitnum = os.path.basename(unitdir).strip(unitname)

    # unit segment processing by indices
    for i, index_range in enumerate(unit_segment_indices):
        # Segment dict
        segment_dict = {}
        # segname = "segment"+str(i+1)
        segname= unitname+unitnum+"-segment"+str(i+1)
        start, end = {}, {}
        start['index'], end['index'] = index_range[0], index_range[1]
        start_lat, start_lon = frames_list[start['index']]['latitude'], frames_list[start['index']]['longitude']
        end_lat, end_lon = frames_list[end['index']]['latitude'], frames_list[end['index']]['longitude']
        start['lrs'] = 0.0001*round(10000*frames_list[start['index']]['lrs'])
        end['lrs'] = 0.0001*round(10000*frames_list[end['index']]['lrs'])
        start['latitude'], start['longitude'] = 1E-8*round(start_lat*1E8), 1E-8*round(1E8*start_lon)
        end['latitude'], end['longitude'] = 1E-8*round(1E8*end_lat), 1E-8*round(1E8*end_lon)
        segment_dict['start'] = start
        segment_dict['end'] = end
        # feature dict
        feature_dict = {
                "type": "Feature",
                "segment_index": seg_num,
                "properties": {
                        "type": "segment",
                        "id": segname,
                        "segment_index": seg_num,
                        "unit": unitname+unitnum,
                        "color": "yellow",
                        "lrs_span": 0.0,
                        "total_frames": 0,
                        "frame_distance": 0.0,
                        "start" :{"index": start['index'], "lrs": start['lrs'], "latitude": start_lat, "longitude": start_lon},
                        "end" :{"index": end['index'], "lrs": end['lrs'], "latitude": end_lat, "longitude": end_lon}
                    },
                "geometry": {
                    "type": "LineString",
                    "width": 4, 
                    "coordinates": [
                    ],
                    "timestamps": [
                    ]
                }
            }
        # frame indices
        i1, i2 = start['index'], min(end['index']+1, len(frames_list))
        coords = []
        for index in range(i1, i2):
            coords.append([frames_list[index]['longitude'], frames_list[index]['latitude']])
        feature_dict['geometry']['coordinates'] = coords
        #
        seg_frame_list = frames_list[i1:i2]
        # get distances for segment
        this_total_frames, this_total_dist = get_seg_dist(seg_frame_list)
        #
        feature_dict['properties']['total_frames'] = this_total_frames
        feature_dict['properties']['frame_distance'] = 0.1*round(10*(this_total_dist))
        # totals
        total_dist = total_dist + this_total_dist
        # total_lrs = total_lrs + this_total_lrs
        total_frames = total_frames + this_total_frames

        # append
        segment_outputs_list.append(segment_dict)
        geojson_dict['features'].append(feature_dict)

        seg_num += 1

    session_summary = {}
    session_summary['audited_frames'] = total_frames
    session_summary['audited_segments'] = seg_num
    geojson_dict['summary'] = session_summary

    # no unit level summarizing

# total frames
print(f"Total segments: {seg_num}, 'Audited frames = {total_frames}, Audited meters = {'%.1f' % total_dist}")


# Individual scores
# summarize for session

summary_dict = {}
# summary_dict['lrs_span'] = 
outputs_json['session'] = summary_dict
outputs_json['segments'] = segment_outputs_list


geojson_object = json.dumps(geojson_dict, indent=4)
with open(aggregate_geofile, "w") as outfile:
    outfile.write(geojson_object)
outfile.close()

print("Created", aggregate_geofile)