#!/usr/bin/env python
import numpy as np

def local2global(x, y, h, x2_local, y2_local):
    x2_global = x2_local*np.cos(h) - y2_local*np.sin(h) + x
    y2_global = y2_local*np.cos(h) + x2_local*np.sin(h) + y
    return x2_global, y2_global

      

        

    
