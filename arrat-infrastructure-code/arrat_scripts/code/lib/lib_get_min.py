#!/usr/bin/env python
import numpy as np

def get_min(input_list):
    # min_value, min_index = get_min(np.abs(np.array(s_time)-item['event_start_time']))
    min_value = np.amin(np.array(input_list))
    return min_value, np.where(input_list == min_value)[0][0]

      

        

    
