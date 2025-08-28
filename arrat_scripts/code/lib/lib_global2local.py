#!/usr/bin/env python
import numpy as np

def global2local(x, y, h, x2, y2):
    x2_local = (x2 - x)*np.cos(h) + (y2 - y)*np.sin(h)
    y2_local = (y2 - y)*np.cos(h) - (x2 - x)*np.sin(h)
    return x2_local, y2_local
      

        

    
