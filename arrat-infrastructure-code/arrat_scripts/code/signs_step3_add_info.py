# Usage: python signs_step3_add_info.py --settings /path/to/inputs.json 

import os, json, argparse, glob
from lib.lib_state_lrs import load_state_lrs
from lib.lib_project_lrs import project_on_lrs

inputs_filename = 'inputs.json'
unitname = 'unit'
use_lrs = False

parser = argparse.ArgumentParser(description="This script accepts one input filepath")
parser.add_argument("--datadir", default="", help="Unit data directory")

args = parser.parse_args()
# ----------- User inputs

datadir = args.datadir
all_units = glob.glob(os.path.join(datadir, unitname+'*'))
glob_sign_num = 0
for unitdir in all_units:
    print('Now processing ', os.path.basename(unitdir))
    session_file = os.path.join(datadir, 'session.geojson')
    output_file = session_file 

    inputs_file = os.path.join(datadir, inputs_filename)
    f = open(inputs_file)
    inputs = json.load(f)
    f.close()

    sign_locator_file = os.path.join(unitdir, inputs['geojson_name'])
    if os.path.exists(sign_locator_file):
        with open(sign_locator_file, 'r') as file:
            sign_data = json.load(file)

        score_bin = inputs['segment_score_bins'] # score_bin = [0.6, 0.9]
        if len(score_bin)==2:
            score_bin.insert(0, 0.0)
            score_bin.insert(3, 1.0)
        # ----------- Process User input files
        if use_lrs:
            lrs_file = inputs['lrs_file']
            state_lat, state_lon, state_lrs, state_nlfid = load_state_lrs(lrs_file)
        
        with open(session_file, 'r') as file:
            session_data = json.load(file)
        # ----------- Process the sign info from the results of sign_locator
        sign_num = 0 
        sign_bin1, sign_bin2, sign_bin3 = 0, 0, 0
        leg_time, conspi_ration, glan_leg, und_max, sign_info, sign_side = [], [], [], [], [], []
        sign_info, sign_lonlat, sign_roi_x, sign_depth, sign_ID, sign_score, sign_lrs, sign_timestamp = [], [], [], [], [], [], [], []
        for i in range(len(sign_data['features'])):
            sign_num+=1
            glob_sign_num+=1
            sign_score.append(sign_data['features'][i]['properties']['overall_score'])
            sign_info.append(sign_data['features'][i]['properties']['info'])
            sign_timestamp.append(sign_data['features'][i]['properties']['filename'])
            sign_roi_x.append(sign_data['features'][i]['properties']['roi_x'])
            sign_depth.append(sign_data['features'][i]['properties']['depth'])
            sign_ID.append(glob_sign_num) #sign_data['features'][i]['properties']['ID']
            leg_time.append(sign_data['features'][i]['properties']['legibility_time'])
            conspi_ration.append(sign_data['features'][i]['properties']['conspicuity'])
            glan_leg.append(sign_data['features'][i]['properties']['glance_legibility'])
            und_max.append(sign_data['features'][i]['properties']['understandability'])
            sign_side.append(sign_data['features'][i]['properties']['sign_side'])
            sign_info.append(sign_data['features'][i]['properties']['info'])
            
            if sign_data['features'][i]['properties']['quality']/100 < score_bin[1]:
                sign_bin1+=1
            elif score_bin[1] <= sign_data['features'][i]['properties']['quality']/100 and sign_data['features'][i]['properties']['quality']/100 < score_bin[2]:
                sign_bin2+=1
            elif score_bin[2] <= sign_data['features'][i]['properties']['quality']/100:
                sign_bin3+=1
            lat,lon = sign_data['features'][i]['geometry']['coordinates'][1], sign_data['features'][i]['geometry']['coordinates'][0]
            sign_lonlat.append([lon, lat ])
            if use_lrs: # Calculate detected sign's LRS value
                this_lrs, pp_lat, pp_lon, valid_all, ci, n1, n2 = project_on_lrs(lat, lon, state_lat, state_lon, state_lrs)
                sign_lrs.append(this_lrs)
            else:
                sign_lrs.append(-1)
        print('Number of speed limit sign: ', sign_num, ' at: ', unitdir)

        if len(sign_score) > 0:

            sign_metrics = {
                "Number of speed limit signs": sign_num,
                "Average score": sum(sign_score)/len(sign_score),
                "score_bins": score_bin,
                "sign_count": [sign_bin1, sign_bin2, sign_bin3]
            }
            # -------  Merge sign data with available laneline info
            for j in range(len(sign_lrs)):
                found = False
                seg_id = 'unitX-segmentX'
                seg_index = 'n/a'
                unit = os.path.basename(unitdir) #'n/a'
                if use_lrs:
                    for i in range(len(session_data['features'])):
                        if session_data['features'][i]['properties']['type'] == 'segment':
                            lrs_start = session_data['features'][i]['properties']['start']['lrs']
                            lrs_end = session_data['features'][i]['properties']['end']['lrs']
                            if session_data['features'][i]['properties']['unit'] == os.path.basename(unitdir): #'unit'+unit_num:
                                if (lrs_start <= sign_lrs[j] and sign_lrs[j] <= lrs_end) or (lrs_start >= sign_lrs[j] and sign_lrs[j] >= lrs_end):
                                    seg_id = session_data['features'][i]['properties']['id']
                                    seg_index = session_data['features'][i]['segment_index']
                                    unit = session_data['features'][i]['properties']['unit']
                                    print('Found! ', sign_lrs[j] , 'is within ', lrs_start, 'and ', lrs_end, 'in', seg_id)
                                    found = True
                                    break
                    if not found:
                        print('Sign_lrs: ', sign_lrs[j], ' was not found in the session file.')

                sign_prop = {
                    "timestamp": sign_timestamp[j],
                    "id": str(sign_ID[j]), # seg_id +"_sign"+ str(sign_ID[j])
                    "unit": unit,
                    "color": 'red',
                    "lrs": sign_lrs[j],
                    "roi_x": sign_roi_x[j],
                    "depth": sign_depth[j],
                    "overall_score": sign_score[j],
                    "legibility_time": leg_time[j],
                    "conspicuity": conspi_ration[j],
                    "glance_legibility": glan_leg[j],
                    "understandability": und_max[j],
                    "side":sign_side[j],
                    "info": sign_info[j]
                }
                sign_dict = {
                    "type": "Feature", 
                    # "segment_index": seg_index,
                    "properties": sign_prop,
                    "geometry": {"type": "Point", "coordinates": sign_lonlat[j]}                       
                }
                session_data['features'].append(sign_dict)
            # -------  Create/ Update Sign metrics for summary
            if 'sign_metrics' in session_data['summary']:
                print('Updating the sign metrics on the session summary')
                metrics = session_data['summary']['sign_metrics']
                if metrics['score_bins'] == sign_metrics['score_bins']:
                    num_old_signs = metrics["Number of speed limit signs"]
                    metrics["Number of speed limit signs"]+=sign_metrics["Number of speed limit signs"]
                    num_new_signs = sign_metrics["Number of speed limit signs"]
                    print('Avg score before: ', metrics['Average score'], '---  num_old_signs: ', num_old_signs, ' num_new_signs: ', num_new_signs)
                    metrics['Average score'] = (num_old_signs*float(metrics['Average score']) + num_new_signs*float(sign_metrics['Average score']))/(num_old_signs+num_new_signs)
                    print('Avg score after: ', metrics['Average score'], '---Total number of signs: ', metrics["Number of speed limit signs"])
                    metrics['sign_count'][0]+=sign_metrics['sign_count'][0]
                    metrics['sign_count'][1]+=sign_metrics['sign_count'][1]
                    metrics['sign_count'][2]+=sign_metrics['sign_count'][2]    
                    print('Successful to update!')
                else:
                    print('Failure to update. score_bins mismatched!')
            else:
                session_data['summary']['sign_metrics'] = sign_metrics
                print('Adding sign metrics to the session summary')

            # ------- Update the existed geojson file
            geojson_object = json.dumps(session_data, indent=4)
            with open(output_file, "w") as outfile:
                outfile.write(geojson_object)

            print("Created ", output_file)

        else:
            print("No signs to add")
    else:
        print("No sign_locator file found in ", unitdir)
        continue