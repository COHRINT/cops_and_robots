import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from shapely.geometry import Polygon,Point,LineString
from shapely.affinity import rotate
from descartes.patch import PolygonPatch

BLUE = '#6699cc'
RED = '#ff3333'
GREEN = '#33ff33'

class MapObj(object):
    """Generate a map object based on a geometric shape, plus 'zones'

        :param name: String
        :param shape: #[(x_i,y_i)] in [m] as a list of positive xy pairs
        :param pose: #[x,y,theta] in [m,m,deg] as the pose of the centroid
        :param shape: #[(x_i,y_i)] in [m] as a list of positive xy pairs
        """
    def __init__(self,name,shape,pose=[0,0,0],has_zones=True):
        self.name = name #sting identifier
        self.has_zones = has_zones
        
        #Define pose as a position and direciton in 2d space
        self.pose = pose #[x,y,theta] in [m] coordinates of the centroid in the global frame
        
        #If shape has only length and width, convert to point-based poly
        if len(shape) == 2:
            shape = [(0,0),(0,shape[1]),(shape[0],shape[1]),(shape[0],0),(0,0)]

        #Define shape as a list of points (assume only solid polygons for now)
        #Construct the shape such that the centroid angle (the direction the object normally faces) is 0
        self.shape = Polygon(shape) #[(x_i,y_i)] in [m] as a list of positive xy pairs 
        x,y = self.shape.centroid.x, self.shape.centroid.y
        shape = [( p[0]-x, p[1]-y ) for p in shape] #place centroid at origin
        self.shape = Polygon(shape)

        #place the shape at the correct pose
        self.move_shape(pose)

    def move_shape(self,pose):
        self.pose = pose
        self.shape = [( p[0]+pose[0], p[1]+pose[1] ) for p in self.shape.exterior.coords]
        self.shape = Polygon(self.shape)
        self.rotate_poly(pose[2],self.shape.centroid)

        self.sides = []
        self.zones = []
        self.zones_by_label = {}
        self.points = self.shape.exterior.coords
        
        #Define zones as areas around the polygons
        if self.has_zones:
            self.define_zones()

    def define_zones(self,zone_distance=0.5):
        resolution = 10
        round_ = 1
        mitre = 2
        bevel = 3

        self.buffer_ = self.shape.buffer(zone_distance,resolution=resolution,join_style=mitre)
        buffer_points = self.buffer_.exterior.coords

        n_sides = len(self.points) - 1 
        n_lines_buffer = len(buffer_points)  - 1
        buffer_lines_per_side = n_lines_buffer / n_sides 
        
        for i,p1 in enumerate(self.points[:-1]):
            p4 = self.points[i+1]

            ps = self.buffer_.exterior.coords[i*buffer_lines_per_side:(i+1)*buffer_lines_per_side+1]
            pts = [p1]
            pts.extend(ps[:])
            pts.extend([p4])
            pts.extend([p1])

            zone = Polygon(pts)
            self.zones.append(zone)

        if n_sides == 4:
            self.zones_by_label['left'] = self.zones[0]
            self.zones_by_label['front'] = self.zones[1]
            self.zones_by_label['right'] = self.zones[2]
            self.zones_by_label['back'] = self.zones[3]

    def rotate_poly(self,angle,rotation_point=(0,0)):
        pts = self.shape.exterior.coords
        lines = []
        for pt in pts:
            line = LineString([rotation_point,pt])
            lines.append(rotate(line,angle,origin=rotation_point))

        pts = []
        for line in lines:
            pts.append(line.coords[1])

        self.shape = Polygon(pts)

    def add_to_plot(self,ax,color=BLUE,include_shape=True,include_zones=False):
        patch = PolygonPatch(self.shape, facecolor=color, alpha=0.5, zorder=2)
        ax.add_patch(patch)

        if include_zones:
            for zone in self.zones:
                patch = PolygonPatch(zone, facecolor=GREEN, alpha=0.5, zorder=2)
                ax.add_patch(patch)

    def __str___(self):
        return "%s is located at (%d,%d), pointing at %d" % (self.name, self.centroid['x'],self.centroid['y'],self.centroid['theta'])

if __name__ == '__main__':
    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width
    shape = [(0,0),(0,w),(l,w),(l,0),(0,0)]
    pose = (2.5,1.5,45)
    wall1 = MapObj('wall1',shape,pose)

    shape = [(0,0),(l/2,w*10),(l,0),(l/2,-w*10),(0,0)]
    pose = (0,0,0)
    wall2 = MapObj('wall2',shape,pose)

    shape = [(0,0),(l/2,w*4),(l,0),(0,0)]
    pose = (2,-2,30)
    wall3 = MapObj('wall3',shape,pose)

    p = Point(0,0)
    circle = p.buffer(1)
    shape = circle.exterior.coords
    pose = (-2.5,-2,30)
    wall4 = MapObj('wall4',shape,pose)


    fig = plt.figure(1,figsize=(10,6)) 
    ax = fig.add_subplot(111)

    wall1.add_to_plot(ax,include_zones=False) 
    wall2.add_to_plot(ax,include_zones=False) 
    wall3.add_to_plot(ax,include_zones=False) 
    wall4.add_to_plot(ax,include_zones=False) 
    
    lim = 5
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    plt.show()      

    fig = plt.figure(1,figsize=(10,6)) 
    ax = fig.add_subplot(111)

    wall1.add_to_plot(ax,include_zones=True) 
    wall2.add_to_plot(ax,include_zones=True) 
    wall3.add_to_plot(ax,include_zones=True) 
    wall4.add_to_plot(ax,include_zones=True) 
    
    lim = 5
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    plt.show()      
