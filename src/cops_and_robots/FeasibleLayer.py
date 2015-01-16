#!/usr/bin/env/python
from matplotlib.colors import cnames
from shapely.geometry import box,Polygon
from cops_and_robots.ShapeLayer import ShapeLayer

class FeasibleLayer(object):
    """A polygon (or collection of polygons) that represent feasible regions of the map. Feasible can be defined as either feasible robot poses or unoccupied space."""
    def __init__(self, bounds, shape_layer=None,visible=True,max_robot_radius = 0.3):
        super(FeasibleLayer, self).__init__()
        self.visible = visible
        self.bounds = bounds #[xmin,ymin,xmax,ymax] in [m]
        self.max_robot_radius = max_robot_radius #[m] conservative estimate of robot size

        self.feasible = None
        self.feasible_pose = None
        if not shape_layer:
            shape_layer = ShapeLayer(bounds)
        print(shape_layer.all_shapes)
        self.define_feasible_regions(shape_layer)        
        

    def define_feasible_regions(self,shape_layer):
        feasible_space = box(*self.bounds)
        for obj_ in shape_layer.shapes.values():
            self.feasible = feasible_space.difference(obj_.shape)

            buffered_shape = obj_.shape.buffer(self.max_robot_radius)
            self.feasible_pose = feasible_space.difference(buffered_shape)

    def plot(self,type_="pose"):
        if type_ is "pose":
            self.feasible_pose.plot()
        else:
            self.feasible.plot()
