import pyproj

def en2latlon(pp, n, e):
    lat, lon = [], []
    for i in range(0,len(e)):
        x, y = pp(e[i], n[i], inverse=True)
        lat.append(y)
        lon.append(x)
    return lat, lon