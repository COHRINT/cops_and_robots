#!/usr/bin/env/python
from matplotlib.colors import cnames
from shapely.geometry import MultiPolygon,Polygon

class ShapeLayer(object):
    """docstring for ShapeLayer"""
    def __init__(self, bounds ,visible=True):
        self.visible = visible
        self.bounds = bounds #[xmin,ymin,xmax,ymax] in [m]
        self.shapes = {} #Dict of MapObj.name : Map object, for each shape
        self.all_shapes = Polygon()

        super(ShapeLayer, self).__init__()
        
    def add_obj(self,map_obj,all_objects=True):
        self.shapes[map_obj.name] = map_obj
        if all_objects:
            self.update_all()

    def rem_obj(self,map_obj_name,all_objects=True):
        del self.shapes[map_obj_name]
        if all_objects:
            self.update_all()

    def update_all(self):
        all_shapes = []
        for obj_ in self.shapes.values():
            all_shapes.append(obj_.shape)
        self.all_shapes = MultiPolygon(all_shapes)

    def plot(self,plot_zones=True,ax=None,alpha=0.8):
        if ax == None:
            ax = plt.gca()
        for shape in self.shapes.values():
            if shape.visible:
                shape.plot(plot_zones=plot_zones,ax=ax,alpha=alpha)