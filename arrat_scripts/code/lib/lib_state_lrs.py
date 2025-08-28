#!/usr/bin/env python
import json

### Load LRS data
def load_state_lrs(lrs_file):
    f = open(lrs_file)
    state_lrs_list = json.load(f)
    f.close()
    state_lat, state_lon, state_lrs, state_nlfid = [], [], [], []
    for element in state_lrs_list:
        state_nlfid.append(element['features'][0]['attributes']['NLFID'])
        state_lat.append(element['features'][0]['geometry']['y'])
        state_lon.append(element['features'][0]['geometry']['x'])
        state_lrs.append(element['features'][0]['attributes']['LogPoint'])
    return state_lat, state_lon, state_lrs, state_nlfid

# def load_state_lrs(lrs_file):
#     f = open(lrs_file)
#     state_lrs_list = json.load(f)
#     f.close()
#     state_lat, state_lon, state_lrs, state_nlfid = [], [], [], []
#     for element in state_lrs_list:
#         state_nlfid.append(element['attributes']['NLFID'])
#         state_lat.append(element['geometry']['y'])
#         state_lon.append(element['geometry']['x'])
#         state_lrs.append(element['attributes']['LogPoint'])
#     return state_lat, state_lon, state_lrs, state_nlfid
