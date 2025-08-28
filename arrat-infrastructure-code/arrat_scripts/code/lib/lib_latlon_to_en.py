#!/usr/bin/env python
import pyproj
def latlon2en(pp, lat, lon):
    e, n = [], []
    for i in range(0, len(lat)):
        x, y = pp(lon[i], lat[i])
        e.append(x)
        n.append(y)
    return e, n