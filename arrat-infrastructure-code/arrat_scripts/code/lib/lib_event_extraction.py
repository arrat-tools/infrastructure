#!/usr/bin/env python
import numpy as np

def moving_median(x, window_size=3):
    x_med = []
    for i in range(0, len(x)):
        if i < window_size:
            x_mov = x[0:i+1]
        else:
            x_mov = x[i-window_size+1:i+1]
        x_med.append(np.median(np.array(x_mov)))
    return x_med

def moving_std(x, window_size=3):
    x_std = []
    for i in range(0, len(x)):
        if i < window_size:
            x_mov = x[0:i+1]
        else:
            x_mov = x[i-window_size+1:i+1]
        x_std.append(np.std(np.array(x_mov)))
    return x_std

def since_thresh(x, val_thresh, age_thresh):
    init = False 
    state, age = [], []
    for i in range(0, len(x)):
        if x[i] > val_thresh:
            i_last_bad = i
            init = True
        if not init:
            age.append(-1)
            state.append(1)
        else:
            if i-i_last_bad >= age_thresh:
                this_state = 1
            else:
                this_state = 0
            age.append(i-i_last_bad)
            state.append(this_state)
    return age, state


def consec_thresh(x, val_thresh, age_thresh):
    init = False 
    state, age = [], []
    for i in range(0, len(x)):
        if x[i] > val_thresh:
            i_last_good = i
            init = True
        if not init:
            age.append(-1)
            state.append(0)
        else:
            if i-i_last_good >= age_thresh:
                this_state = 0
            else:
                this_state = 1
            age.append(i-i_last_good)
            state.append(this_state)
    for i in range(age_thresh-1, len(state)):
        if state[i] == 0:
            state[i-age_thresh+1:i] = [0] * (age_thresh-1)
    return age, state

      

        

    
