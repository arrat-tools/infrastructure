#!/usr/bin/env python
import json, os, sys, copy, argparse
import numpy as np
np.float = float
from lib.lib_segment_scores import segment_laneline_visibility, segment_laneline_consistency, segment_laneline_detection, segment_laneline_curvature, segment_laneline_combined

# python frames_to_segments.py --settings /path/to/inputs.json

inputs_filename = 'inputs.json'
unitname = 'unit'
frames_filename = 'frames.json'
session_filename = "session.geojson"
mile2meter = 1609.34
score_colors = ['red', 'yellow', 'green']

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

# Score bins
init_summary = False

# Open aggregate geojson
f = open(aggregate_geofile)
session_json = json.load(f)
f.close()

total_frames = 0
total_dist = 0.0

output_json = session_json

output_features = []

# run through all segments from geojson features list
for i, seg_dict in enumerate(session_json['features']):

    if seg_dict['properties']['type'] == 'segment':

        # open unit file for segment
        unit = seg_dict['properties']['unit']
        unitdir = os.path.join(datadir, unit)

        # Open inputs file
        inputs_file = os.path.join(unitdir, inputs_filename)
        f = open(inputs_file)
        inputs = json.load(f)
        f.close()

        frames_filename = inputs['frames_filename']
        frames_file = os.path.join(unitdir, frames_filename)
        f = open(frames_file)
        frames_list = json.load(f)['laneline_frames']
        f.close()

        
        # Initialize
        if not init_summary:
            init_summary = True

            segment_score_bins = inputs['segment_score_bins']
            score_thresh = [0.0]
            for bin in segment_score_bins:
                score_thresh.append(bin)
            score_thresh.append(1.0)
            # metric functions dict
            metric_functions_dict = {'laneline_visibility': 'segment_laneline_visibility',
                                    'laneline_consistency': 'segment_laneline_consistency',
                                    'laneline_detection': 'segment_laneline_detection',
                                    'laneline_combined': 'segment_laneline_combined'}

            # make placeholder data for metric summary
            metric_summary_data = {'bad_frames':0, 
                                'bad_dist':0.0, 
                                'score':0.0, 
                                'seg_num_hist': [0] * (len(score_thresh)-1), 
                                'seg_dist_hist': [0.0] * (len(score_thresh)-1)}
            metric_summary_list = []
            for metric_name in metric_functions_dict.keys():
                this_data = copy.deepcopy(metric_summary_data)
                this_dict = {'name': metric_name, 'data': this_data}
                metric_summary_list.append(this_dict)

        # create empty metrics key for segment
        seg_dict['laneline_metrics'] = []

        total_frames = total_frames + seg_dict['properties']['total_frames']
        segment_dist = seg_dict['properties']['frame_distance']
        total_dist = total_dist + segment_dist

        # frame indices for segment
        i1, i2 = seg_dict['properties']['start']['index'], seg_dict['properties']['end']['index']
        segment_frames = frames_list[i1:i2+1]

        # Individual segment all metric calcs
        this_bad_segment_dist = 0.0
        this_segment_denominator_dist = 0.0
        for metric in metric_summary_list:
            # metric_name = 'laneline_visibility'
            metric_name = metric['name']
            this_bad_count, this_bad_dist = eval(metric_functions_dict[metric_name] + "(segment_frames)")
            this_bad_segment_dist = this_bad_segment_dist + this_bad_dist
            this_segment_denominator_dist = this_segment_denominator_dist + segment_dist
            metric['data']['bad_frames'] = metric['data']['bad_frames'] + this_bad_count
            metric['data']['bad_dist'] = metric['data']['bad_dist'] + this_bad_dist
            segment_score = 1.0 - max(0.0, min(this_bad_dist/segment_dist, 1.0))
            for iThresh in range(0, len(score_thresh)-1):
                if segment_score >= score_thresh[iThresh] and segment_score <= score_thresh[iThresh+1]:
                    metric['data']['seg_num_hist'][iThresh] = metric['data']['seg_num_hist'][iThresh] + 1
                    metric['data']['seg_dist_hist'][iThresh] = metric['data']['seg_dist_hist'][iThresh] + segment_dist

                    # segment score based color
                    if metric_name == 'laneline_combined':
                        seg_dict['properties']['color'] = score_colors[iThresh]


            seg_dict['laneline_metrics'].append({'name': metric_name, 'score': segment_score})
        segment_laneline_score = 1.0 - max(0.0, min(this_bad_segment_dist/this_segment_denominator_dist, 1.0))

        output_features.append(seg_dict) # segment data is ro be appended to geojson features list

    else:
        output_features.append(seg_dict)

# assemble summary for all metrics
metric_summary_ouput = []
for metric in metric_summary_list:
    metric_name = metric['name']
    session_score = 1.0 - max(0.0, min(metric['data']['bad_dist']/total_dist, 1.0))
    
    # make a combined lane line score variable to combine with signs for overall score
    if metric_name == 'laneline_combined':
        laneline_combined_score = session_score
    #

    metric_summary_dict = {'name': metric_name, 
                'score': session_score, 
                'distribution': 
                        {'score_bins': score_thresh, 
                        'segment_count': metric['data']['seg_num_hist'], 
                        'segment_dist': metric['data']['seg_dist_hist']
                        }
            }
    metric_summary_ouput.append(metric_summary_dict)

# get overall lane line combined score for summary 
# to do - combine with signs score

signs_combined_score = laneline_combined_score
overall_score = 1/2*(laneline_combined_score + signs_combined_score)
 
output_json['features'] = output_features
output_json['summary']['overall_score'] = overall_score
output_json['summary']['laneline_metrics'] = metric_summary_ouput
geojson_object = json.dumps(output_json, indent=4)
with open(aggregate_geofile, "w") as outfile: # aggregate_geofile
    outfile.write(geojson_object)
outfile.close()