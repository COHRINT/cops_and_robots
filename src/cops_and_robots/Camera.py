from shapely.geometry import MultiPolygon,Polygon,Point,box
from shapely import affinity
from cops_and_robots.MapObj import MapObj
import math
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch



#sublcass of MapObj
class Camera(MapObj):
    """docstring for Camera"""
    def __init__(self):
        
        #Define nominal viewcone
        self.min_view_dist = 0.3 #[m]
        self.max_view_dist = 1.0 #[m]
        self.view_angle = math.pi/2 #[rad]
        viewcone_pts = [
                        (0,0), 
                        (self.max_view_dist * math.cos(self.view_angle/2), self.max_view_dist * math.sin(self.view_angle/2)),
                        (self.max_view_dist * math.cos(-self.view_angle/2), self.max_view_dist * math.sin(-self.view_angle/2)),
                        (0,0),
        ]

        self.viewcone = Polygon(viewcone_pts) #this is the unencumbered viewcone. self.shape represents the actual view
        self.view_pt = (0,0)
        self.offset = (0,0)#(-0.1,-0.1) #[m] offset (x,y) from center of robot
        
        #Define simlated sensor parameters
        self.update_freq = 1 #[hz}
        self.detection_chance = 0.9 # P(detect|x) where x is in the viewcone

        #Instantiate MapObj superclass object
        name = "Camera"
        super(Camera, self).__init__(name,shape_pts=viewcone_pts,pose=[0,0,0],has_zones=False,centroid_at_origin=False)
     
    def update(self,robot_pose,map_):
           self.move_viewcone(robot_pose)
           self.rescale_viewcone(robot_pose,map_)   

    def move_viewcone(self,robot_pose):
        """Move the viewcone based on the robot's pose"""
        pose = (
                robot_pose[0] + self.offset[0],  
                robot_pose[1] + self.offset[1],
                robot_pose[2]
        )

        self.shape = self.viewcone
        self.viewcone = self.move_shape(pose,rotation_pt=self.view_pt)
        self.view_pt = self.shape.exterior.coords[0]

    def rescale_viewcone(self,robot_pose,map_):
        if self.shape.intersects(map_):
            # origin = self.shape.project(map_object.shape)#map_object.shape.exterior.coords[1]
            # shadow = affinity.scale(map_object.shape,xfact=1000,yfact=1000,origin=origin) #map portion invisible to the viewcone
            # self.shape = self.shape.difference(shadow)
            distance = Point(self.view_pt).distance(map_)
            scale = distance/self.max_view_dist*1.3 #<>TODO: why the 1.3 multiplier?
            self.shape = affinity.scale(self.viewcone,xfact=scale,yfact=scale,origin=self.view_pt)
        else:
            self.shape = self.viewcone
                

    def sensor_model(self):
        """ """
        pass

    def check_detection(self,target):
        """ Update camera once per update period """
        pass

if __name__ == '__main__':

    BLUE = '#6699cc'
    RED = '#ff3333'
    GREEN = '#33ff33'

    #Run a test to make sure camera's working
    kinect = Camera()

    fig = plt.figure(1,figsize=(8,8)) 
    ax = fig.add_subplot(111)

    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width

    pose = (2.4,0,90)
    wall1 = MapObj('wall1',(l,w),pose)
    wall1.add_to_plot(ax,include_zones=False)
    # origin = wall1.shape.exterior.coords[1]
    # shadow = affinity.scale(wall1.shape,xfact=1000,yfact=1000,origin=origin) #map portion invisible to the viewcone
    # patch = PolygonPatch(shadow, facecolor=RED, alpha=0.5, zorder=2)
    # ax.add_patch(patch)

    pose = (2.5,2.5,45)
    wall2 = MapObj('wall2',(l,w),pose)
    wall2.add_to_plot(ax,include_zones=False)

    map_ = MultiPolygon((wall1.shape,wall2.shape))
    
    goal_points = [(0,0,0),(2,0,0),(0,0.5,90),(0,1,0),(0,0.2,0)]

    for point in goal_points:
        kinect.update(point,map_)
        kinect.add_to_plot(ax,include_zones=False,color=GREEN)  
        # patch = PolygonPatch(kinect.viewcone, facecolor=RED, alpha=0.5, zorder=2)
        # ax.add_patch(patch)   

    
    lim = 5
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])

    plt.show()
    