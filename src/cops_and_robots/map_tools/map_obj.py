#!/usr/bin/env python
"""Defines physical and non-physical objects used in the map environment.

``map_obj`` extends Shapely's geometry objects (generally polygons) to 
be used in a robotics environmnt. Map objects can be physical, 
representing walls, or non-physical, representing camera viewcones.

The visibility of an object can be toggled, and each object can have 
*zones* which define areas around the object. For example, a 
rectangular wall has four zones: front, back, left and right. These 
are named zones, but arbitrary shapes can have arbitrary numbered zones 
(such as a triangle with three numbered zones).

Required Knowledge:
    This module and its classes do not need to know about any other 
    parts of the cops_and_robots parent module.
"""

__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib.path import Path
from shapely.geometry import box,Polygon,Point,LineString
from shapely.affinity import rotate
from descartes.patch import PolygonPatch

class MapObj(object):
    """Generate an object based on a geometric shape, plus 'zones' 
    which demarcate spatial relationships around objects.

    Note:
        If only one xy pair is given as shape_pts, MapObj will assume 
        the user wants to create a box with those two values as length
        and width, respectively.

    Note:
        Shapes are created such that the centroid angle (the direction
        the object is facing) is 0. To change this, use ``move_shape``.

    :param name: the map object's name.
    :type name: String.
    :param shape_pts: a list of xy pairs as [(x_i,y_i)] in [m,m] in the 
        global (map) frame of reference.
    :type shape_pts: an n-element list of 2-element lists of floats.
    :param pose: the centroid's pose as [x,y,theta] in [m,m,deg].
    :type pose: a 3-element list of floats.
    :param has_zones: whether or not an object has zones.
    :type has_zones: bool.
    :param centroid_at_origin: place centroid at the origin (instead 
        of the minimum point given by the shape_pts).
    :type centroid_at_origin: bool.
    :param visible: whether or not to show the object on the map.
    :type visible: bool.
    :param default_color_str: string to define the object's default 
        color.
    :type default_color_str: String.
    """

    def __init__(self,name,shape_pts,pose=[0,0,0],has_zones=True,
                 centroid_at_origin=True,visible=True,
                 default_color_str='darkblue',**kwargs):
        #Define basic MapObj properties
        self.name = name 
        self.visible = visible
        self.has_zones = has_zones
        self.default_color = cnames[default_color_str]
        self.pose = pose
        
        #If shape has only length and width, convert to point-based poly
        if len(shape_pts) == 2:
            shape_pts = [list(b) for b in 
                         box(0,0,shape_pts[0],shape_pts[1]).exterior.coords]

        #Build the map object's polygon (shape)
        if centroid_at_origin:
            shape = Polygon(shape_pts) 
            x,y = shape.centroid.x, shape.centroid.y
            shape_pts = [( p[0]-x, p[1]-y ) for p in shape_pts] 
        self.shape = Polygon(shape_pts)

        #Place the shape at the correct pose
        self.move_shape(pose)

    def move_shape(self,pose,rotation_pt=None):
        """Translate and rotate the shape.

        The rotation is assumed to be about the object's centroid 
        unless a rotation point is specified.

        :param pose: the centroid's pose as [x,y,theta] in [m,m,deg].
        :type pose: a 3-element list of floats.
        :param rotation_pt: the rotation point as [x,y] in [m,m].
        :type rotation_pt: a 2-element list of floats.
        """
        if rotation_pt:
            rotation_point = rotation_pt
        else:
            rotation_point = self.shape.centroid
        
        #Rotate the polygon
        self.rotate_poly(pose[2],rotation_point)

        #Translate the polygon
        self.pose = pose
        shape_pts = [( p[0]+pose[0], p[1]+pose[1] ) 
                     for p in self.shape.exterior.coords]
        self.shape = Polygon(shape_pts)

        #Redefine sides, points and and zones
        self.points = self.shape.exterior.coords
        self.sides = []
        self.zones = []
        self.zones_by_label = {}
        if self.has_zones:
            self.define_zones()

    def rotate_poly(self,angle,rotation_point):
        """Rotate the shape about a rotation point.

        :param pose: angle to be rotated in degrees.
        :type pose: float.
        :param rotation_pt: the rotation point as [x,y] in [m,m].
        :type rotation_pt: a 2-element list of floats.
        """
        pts = self.shape.exterior.coords
        lines = []
        for pt in pts:
            line = LineString([rotation_point,pt])
            lines.append(rotate(line,angle,origin=rotation_point))

        pts = []
        for line in lines:
            pts.append(line.coords[1])

        self.shape = Polygon(pts)

    def define_zones(self,zone_distance=0.5,resolution=10,join_style='mitre'):
        """Define the shape's zones at a given distance.

        Define areas near the shape ('zones') which, for a four-sided 
        shape, demarcate front, back left and right.

        :param zone_distance: zone distance from the shape's outermost 
            edge.
        :type zone_distance: positive float.
        :param resolution: the resolution of the buffered zone.
        :type resolution: positive float.
        :param join_style: style of the buffered zone creation, taken 
            as one of:
                1. round
                2. mitre
                3. bevel
        :type join_style: string.
        """
        #Create the buffer around the object
        join_styles = ['round','mitre','bevel']
        join_style = join_styles.index(join_style) + 1

        self.buffer_ = self.shape.buffer(zone_distance,resolution=resolution,
                                         join_style=join_style)
        buffer_points = self.buffer_.exterior.coords

        #Prepare to divide the buffer
        n_sides = len(self.points) - 1 
        n_lines_buffer = len(buffer_points)  - 1
        buffer_lines_per_side = n_lines_buffer / n_sides 
        
        #Divide the buffer into specific zones
        for i,p1 in enumerate(self.points[:-1]):
            p4 = self.points[i+1]

            ps = self.buffer_.exterior.coords[i*buffer_lines_per_side:
                                              (i+1)*buffer_lines_per_side+1]
            pts = [p1]
            pts.extend(ps[:])
            pts.extend([p4])
            pts.extend([p1])

            zone = Polygon(pts)
            self.zones.append(zone)

        #Generate labeled zones for 4-sided shapes
        if n_sides == 4:
            self.zones_by_label['left'] = self.zones[0]
            self.zones_by_label['front'] = self.zones[1]
            self.zones_by_label['right'] = self.zones[2]
            self.zones_by_label['back'] = self.zones[3]

    def plot(self,plot_zones=False,ax=None,alpha=0.5,**kwargs):
        """Plot the map_object as a polygon patch.

        Note: 
            The zones can be plotted without the shape if the shape's 
            ``visible`` attribute is False, but ``plot_zones`` is True.

        :param plot_zones:
        :type plot_zones: bool.
        :param ax: Axes to be used for plotting the shape.
        :type ax: Axes.
        :param alpha: Transparency of all elements of the shape.
        :type alpha: float.
        :returns: the polygon as a patch.
        :rtype: PolygonPatch.
        """
        if ax == None:
            ax = plt.gca()
            
        patch = PolygonPatch(self.shape,facecolor=self.default_color,
                             alpha=alpha,zorder=2,**kwargs)
        ax.add_patch(patch)

        if plot_zones:
            for zone in self.zones:
                zone_patch = PolygonPatch(zone,facecolor=cnames['lightgreen'], 
                                          alpha=alpha,zorder=2)
                ax.add_patch(zone_patch)

        return patch

    def __str___(self):
        str_ = "{} is located at ({},{}), pointing at {}}"
        return str_.format(
                           self.name,
                           self.centroid['x'],
                           self.centroid['y'],
                           self.centroid['theta'],
                           )

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
    
    lim = 5
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    plt.show()      
