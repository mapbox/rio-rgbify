import math

mercator_a = 6378137.0
mercator_max = 20037508.342789244

rad = 0.017453292519943295
a = 6378245.0
ee = 0.00669342162296594323


def mercator_to_wgs(x, y):
    lon = x / rad / mercator_a
    lat = (math.pi * 0.5 - 2.0 * math.atan(math.exp(-y / mercator_a))) / rad

    return [lon, lat]


def wgs_to_mercator(lon, lat):
    adjusted = lon if abs(lon) <= 180 else lon - _sign(lon) * 360

    x = mercator_a * adjusted * rad
    y = mercator_a * math.log(math.tan(math.pi * 0.25 + 0.5 * lat * rad))

    if (x > mercator_max):
        x = mercator_max
    if (x < -mercator_max):
        x = -mercator_max
    if (y > mercator_max):
        y = mercator_max
    if (y < -mercator_max):
        y = -mercator_max

    return [x, y];


def gcj_to_wgs(lon, lat):
    if _out_of_china(lon, lat):
        return [lon, lat]
    dlat = _transformlat(lon - 105.0, lat - 35.0)
    dlon = _transformlon(lon - 105.0, lat - 35.0)
    radlat = lat * rad
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = dlat / ((a * (1 - ee)) / (magic * sqrtmagic)* rad)
    dlon = dlon / (a / sqrtmagic * math.cos(radlat)* rad)
    return [lon - dlon, lat - dlat]


def _sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def _out_of_china(lon, lat):
    return not (lon > 73.66 and lon < 135.05 and lat > 3.86 and lat < 53.55)


def _transformlat(lon, lat):
    ret = -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lon * lat + 0.2 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 *
            math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 *
            math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 *
            math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlon(lon, lat):
    ret = 300.0 + lon + 2.0 * lat + 0.1 * lon * lon + \
          0.1 * lon * lat + 0.1 * math.sqrt(abs(lon))
    ret += (20.0 * math.sin(6.0 * lon * math.pi) + 20.0 *
            math.sin(2.0 * lon * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lon * math.pi) + 40.0 *
            math.sin(lon / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lon / 12.0 * math.pi) + 300.0 *
            math.sin(lon / 30.0 * math.pi)) * 2.0 / 3.0
    return ret
