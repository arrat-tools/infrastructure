#!/usr/bin/env python
import json, pyproj
import numpy as np
from shapely.geometry import Point
from shapely.geometry import LineString

""" Lat Long to East North """
_e = 0.0818191908426
_R = 6378137

class LL_NE(object):
    def __init__(self, refLat=0, refLong=0):
        self.update(refLat, refLong)

    def update(self, refLat, refLong):
        self.RefLat = refLat
        self.RefLong = refLong
        self.EN_factors(refLat)

    def EN_factors(self, RefLat):
        self.eFactor = np.cos(RefLat*np.pi/180)*_R/np.sqrt(1-(np.sin(RefLat*np.pi/180)**2*_e**2))*np.pi/180
        self.nFactor = (1-_e**2)*_R/((1-(np.sin(RefLat*np.pi/180)**2*_e**2))*np.sqrt(1-(np.sin(RefLat*np.pi/180)**2*_e**2)))*np.pi/180

    def LL2NE(self, latitude, longitude):
        pos_east = (longitude - self.RefLong) * self.eFactor
        pos_north = (latitude - self.RefLat) * self.nFactor
        return pos_north, pos_east

    def NE2LL(self, pos_north, pos_east):
        longitude = (pos_east/self.eFactor) + self.RefLong 
        latitude = (pos_north/self.nFactor) + self.RefLat
        return latitude, longitude

def project_point(p1, p2, p):
    
    # projection
    point = Point(p[0], p[1])
    line = LineString([p1, p2])
    x = np.array(point.coords[0])
    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])
    n = v - u
    n /= np.linalg.norm(n, 2)
    P = u + n*np.dot(x - u, n)
    pp = (P[0], P[1])
    
    # distance
    projected_d = np.hypot(p[0]-pp[0], p[1]-pp[1])
    
    # inside?
    x_in = (pp[0] >= p1[0] and pp[0] <= p2[0]) or (pp[0] <= p1[0] and pp[0] >= p2[0])
    y_in = (pp[1] >= p1[1] and pp[1] <= p2[1]) or (pp[1] <= p1[1] and pp[1] >= p2[1])
    
    # location
    if x_in or y_in:
        total_d = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        d1 = np.hypot(pp[0]-p1[0], pp[1]-p1[1])
        if total_d == 0.0:
            location = -1
        else:
            location = d1/total_d
    else:
        location = -1
    
    return pp, x_in and y_in, location, projected_d

def project_on_lrs(lat, lon, lat_list, lon_list, lrs_list):
    llne = LL_NE(lat, lon)
    dist_ll = []
    for i in range(0, len(lat_list)):
        dist_ll.append(np.hypot(lat_list[i]-lat, lon_list[i]-lon))
    ci = np.where(dist_ll == np.amin(dist_ll))[0][0]

    # get indices of plus and minus because adhacent may be the same
    if abs(lrs_list[ci]-lrs_list[ci-1]) < 0.05:
        i_minus = ci - 2
    else:
        i_minus = ci - 1
    if abs(lrs_list[ci]-lrs_list[ci+1]) < 0.05:
        i_plus = ci + 2
    else:
        i_plus = ci + 1

    # check if inside behind or ahead
    # start with ahead (plus)
    y1, x1 = llne.LL2NE(lat_list[ci], lon_list[ci])
    y2, x2 = llne.LL2NE(lat_list[i_plus], lon_list[i_plus])
    inside_dir = []
    location_dir = []
    pp, inside, location, d = project_point((x1, y1), (x2, y2), (0.0, 0.0))
    inside_dir.append(inside)
    location_dir.append(location)
    if inside:
        lrs1, lrs2 = lrs_list[ci], lrs_list[i_plus]
        n1, n2 = ci, i_plus
    else: # check behind (minus)
        y2, x2 = llne.LL2NE(lat_list[i_minus], lon_list[i_minus])
        pp, inside, location, d = project_point((x1, y1), (x2, y2), (0.0, 0.0))
        inside_dir.append(inside)
        location_dir.append(location)
        if inside:
            lrs1, lrs2 = lrs_list[ci], lrs_list[i_minus]
            n1, n2 = ci, i_minus
        else:
            y1, x1 = llne.LL2NE(lat_list[i_minus], lon_list[i_minus])
            y2, x2 = llne.LL2NE(lat_list[i_plus], lon_list[i_plus])
            pp, inside, location, d = project_point((x1, y1), (x2, y2), (0.0, 0.0))
            lrs1, lrs2 = lrs_list[i_minus], lrs_list[i_plus]
            n1, n2 = i_minus, i_plus
            inside_dir.append(inside)
            location_dir.append(location)
    pp_lat, pp_lon = llne.NE2LL(pp[1], pp[0])
    this_lrs = lrs1 + (lrs2-lrs1)*location
    return this_lrs, pp_lat, pp_lon, inside_dir, ci, n1, n2
