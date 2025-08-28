#!/usr/bin/env python
import numpy as np

def map2veh(xv, yv, hv, xm, ym):
    x = (xm - xv)*np.cos(hv) + (ym - yv)*np.sin(hv)
    y = (ym - yv)*np.cos(hv) - (xm - xv)*np.sin(hv)
    return x, y

def veh2map(xv, yv, hv, x, y):
    xm = x*np.cos(hv) - y*np.sin(hv) + xv
    ym = y*np.cos(hv) + x*np.sin(hv) + yv
    return xm, ym

      

        

    
