import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib.path import Path
from shapely.geometry import Polygon,Point,LineString
from shapely.affinity import rotate
from descartes.patch import PolygonPatch

class MapObj(object):
    """Generate a map object based on a geometric shape, plus 'zones'

        :param name: String
        :param shape: [(x_i,y_i)] in [m] as a list of positive xy pairs
        :param pose: [x,y,theta] in [m,m,deg] as the pose of the centroid
        :param shape: [(x_i,y_i)] in [m] as a list of positive xy pairs
        """
    def __init__(self,name,shape_pts,pose=[0,0.5,0],has_zones=True,centroid_at_origin=True,visible=True,default_color=None):
        self.visible = visible
        self.name = name #sting identifier
        self.has_zones = has_zones
        self.default_color = default_color
        
        #Define pose as a position and direciton in 2d space
        self.pose = pose #[x,y,theta] in [m] coordinates of the centroid in the global frame
        
        #If shape has only length and width, convert to point-based poly
        if len(shape_pts) == 2:
            shape_pts = [(0,0),(0,shape_pts[1]),(shape_pts[0],shape_pts[1]),(shape_pts[0],0),(0,0)]

        #Define shape as a list of points (assume only solid polygons for now)
        #Construct the shape such that the centroid angle (the direction the object normally faces) is 0
        if centroid_at_origin:
            shape = Polygon(shape_pts) #[(x_i,y_i)] in [m] as a list of positive xy pairs 
            x,y = shape.centroid.x, shape.centroid.y
            shape_pts = [( p[0]-x, p[1]-y ) for p in shape_pts] #place centroid of shape at origin (instead of minimum point)
        self.shape = Polygon(shape_pts)

        #place the shape at the correct pose
        self.move_shape(pose)

    def move_shape(self,pose,rotation_pt="centroid"):
        """Translate and rotate the shape

        :param pose: [x,y,theta] in [m,m,deg] as the pose of the centroid
        """
        if rotation_pt is "centroid":
            rotation_point = self.shape.centroid
        else:
            rotation_point = rotation_pt
        self.rotate_poly(pose[2],rotation_point)

        self.pose = pose
        shape_pts = [( p[0]+pose[0], p[1]+pose[1] ) for p in self.shape.exterior.coords]
        self.shape = Polygon(shape_pts)

        self.sides = []
        self.zones = []
        self.zones_by_label = {}
        self.points = self.shape.exterior.coords
        
        #Define zones as areas around the polygons
        if self.has_zones:
            self.define_zones()

        return Polygon(shape_pts)

    def rotate_poly(self,angle,rotation_point):
        pts = self.shape.exterior.coords
        lines = []
        for pt in pts:
            line = LineString([rotation_point,pt])
            lines.append(rotate(line,angle,origin=rotation_point))

        pts = []
        for line in lines:
            pts.append(line.coords[1])

        self.shape = Polygon(pts)

    def define_zones(self,zone_distance=0.5):
        """Define areas near the shape ('zones') which, for a four-sided shape, demarcate front, back left and right

        :param zone_distance: Double in [m] from the outermost edge of the shape
        """

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

    def plot(self,color=cnames['darkblue'],plot_shape=True,plot_zones=False,alpha=0.5,ax=None):
        if not self.default_color == None:
            color = self.default_color

        if ax == None:
            ax = plt.gca()
            
        patch = PolygonPatch(self.shape, facecolor=color, alpha=alpha, zorder=2)
        ax.add_patch(patch)

        if plot_zones:
            for zone in self.zones:
                zone_patch = PolygonPatch(zone, facecolor=cnames['lightgreen'], alpha=alpha, zorder=2)
                ax.add_patch(zone_patch)

        return patch

    def __str___(self):
        return "%s is located at (%d,%d), pointing at %d" % (self.name, self.centroid['x'],self.centroid['y'],self.centroid['theta'])

if __name__ == '__main__':
    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width

    fig = plt.figure(1,figsize=(10,6)) 
    ax = fig.add_subplot(111)

    shape = [(0,0),(3*l,w),(3*l,-w),(0,0)]
    pose = (2,-2,0)
    wall1 = MapObj('wall3',shape,pose)

    wall1.plot(plot_zones=False) 

    wall1.move_shape((0,0,90),wall1.shape.exterior.coords[0])

    wall1.plot(color=cnames['darkred'],plot_zones=False) 


    # shape = [(0,0),(0,w),(l,w),(l,0),(0,0)]
    # pose = (2.5,1.5,45)
    # wall1 = MapObj('wall1',shape,pose)

    # shape = [(0,0),(l/2,w*10),(l,0),(l/2,-w*10),(0,0)]
    # pose = (0,0,0)
    # wall2 = MapObj('wall2',shape,pose)

    # shape = [(0,0),(l/2,w*4),(l,0),(0,0)]
    # pose = (2,-2,30)
    # wall3 = MapObj('wall3',shape,pose)

    # p = Point(0,0)
    # circle = p.buffer(1)
    # shape = circle.exterior.coords
    # pose = (-2.5,-2,30)
    # wall4 = MapObj('wall4',shape,pose)


    # fig = plt.figure(1,figsize=(10,6)) 
    # ax = fig.add_subplot(111)

    # wall1.plot(plot_zones=False) 
    # wall2.plot(plot_zones=False) 
    # wall3.plot(plot_zones=False) 
    # wall4.plot(plot_zones=False) 
    
    # lim = 5
    # ax.set_xlim([-lim,lim])
    # ax.set_ylim([-lim,lim])
    # plt.show()      

    # fig = plt.figure(1,figsize=(10,6)) 
    # ax = fig.add_subplot(111)

    # wall1.plot(plot_zones=True) 
    # wall2.plot(plot_zones=True) 
    # wall3.plot(plot_zones=True) 
    # wall4.plot(plot_zones=True) 
    
    lim = 5
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    plt.show()      
