from lxml import etree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon, box
from shapely.prepared import prep
from shapely.ops import transform
from functools import partial
from pysal.esda.mapclassify import Natural_Breaks as nb
from descartes import PolygonPatch
import fiona
import pyproj
from itertools import chain

shp = fiona.open('denver-boulder/denver-boulder_colorado_osm_waterways.shp')
bds = shp.bounds
shp.close()

# p_in = pyproj.Proj(shp.crs)
# bound_box = box(*shp.bounds)
# p_out = pyproj.Proj({'init': 'EPSG:4326'})  # aka WGS84
# project = partial(pyproj.transform, p_in, p_out)
# bound_box_wgs84 = transform(project, bound_box)
# ur = bound_box_wgs84.exterior.coords[2]
# ll = bound_box_wgs84.exterior.coords[3]

extra = 0.01
ll = (bds[0], bds[1])
ur = (bds[2], bds[3])
coords = list(chain(ll, ur))
coords = [c/1000 for c in coords]
w, h = coords[2] - coords[0], coords[3] - coords[1]


m = Basemap(
    projection='tmerc',
    lon_0=105.,
    lat_0=40.,
    ellps = 'WGS84',
    llcrnrlon=coords[0] - extra * w,
    llcrnrlat=coords[1] - extra + 0.01 * h,
    urcrnrlon=coords[2] + extra * w,
    urcrnrlat=coords[3] + extra + 0.01 * h,
    lat_ts=0,
    resolution='i',
    suppress_ticks=True)
m.readshapefile(
    'denver-boulder/denver-boulder_colorado_osm_waterways',
    'denver',
    color='none',
    zorder=2)

for k in m.denver_info[:100]:
    print k
# set up a map dataframe
df_map = pd.DataFrame({
    'poly': [Polygon(xy) for xy in m.denver],
    'ward_name': [ward['NAME'] for ward in m.denver_info]})
df_map['area_m'] = df_map['poly'].map(lambda x: x.area)
df_map['area_km'] = df_map['area_m'] / 100000

# Create Point objects in map coordinates from dataframe lon and lat values
map_points = pd.Series(
    [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in zip(df['lon'], df['lat'])])
plaque_points = MultiPoint(list(map_points.values))
wards_polygon = prep(MultiPolygon(list(df_map['poly'].values)))
# calculate points that fall within the London boundary
ldn_points = filter(wards_polygon.contains, plaque_points)