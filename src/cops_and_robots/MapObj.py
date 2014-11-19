import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


class MapObj(object):
    """Generate one or more probability and occupancy layer

        :param target_name: String
        """
    def __init__(self,name,shape,pose=[0,0,0],centroid=[0,0,0],has_poly=0):
        self.name = name #sting identifier
        self.shape = shape #[l,w] in [m] length and width of a rectangular object   
        #self.shape      = shape     #nx2 array containing all (x,y) vertices for the object shape
        self.pose = pose #[x,y,theta] in [m] coordinates of the centroid in the global frame
        self.centroid = centroid  #(x,y,theta) [m] coordinates of the centroid in the object frame
        self.has_poly = has_poly

        #Relative polygons
        if has_poly:
            l,w = (shape[0], shape[1])
            spread = [1.5, 5] #trapezoidal spread from anchor points
            if l > w: 
                x = [-l/2, -spread[0]*l/2, spread[0]*l/2, l/2, -l/2]
                x = [i + pose[0] for i in x]
                y = [w/2, spread[1]*w, spread[1]*w, w/2, w/2]
                y = [i + pose[1] for i in y]
            else:
                y = [-w/2, -spread[0]*w/2, spread[0]*w/2, w/2, -w/2]
                y = [i + pose[1] for i in y]
                x = [l/2, spread[1]*l, spread[1]*l, l/2, l/2]
                x = [i + pose[0] for i in x]
            self.front_poly = np.array([list(i) for i in list(zip(x,y))])
            self.back_poly = []
            self.left_poly = []
            self.right_poly = []
        else:
            self.front_poly = np.array([])
            self.back_poly = []
            self.left_poly = []
            self.right_poly = []

    def __str___(self):
        return "%s is located at (%d,%d), pointing at %d" % (self.name, self.centroid['x'],self.centroid['y'],self.centroid['theta'])
