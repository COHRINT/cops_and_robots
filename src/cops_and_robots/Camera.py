import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import cnames
from descartes.patch import PolygonPatch
from shapely.geometry import MultiPolygon,Polygon,Point,box
from shapely import affinity
from cops_and_robots.MapObj import MapObj
from cops_and_robots.ParticleFilter import *
from cops_and_robots.Map import Map

class Camera(MapObj):
    """docstring for Camera"""
    def __init__(self,robot_pose=(0,0,0),visible=True,default_color=cnames['yellow']):
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

        #Instantiate MapObj superclass object
        name = "Camera"
        super(Camera, self).__init__(name,shape_pts=viewcone_pts,pose=[0,0,0],has_zones=False,centroid_at_origin=False,default_color=default_color)


        self.viewcone = Polygon(viewcone_pts) #this is the unencumbered viewcone. self.shape represents the actual view
        self.view_pose = (0,0,0)
        self.offset = (0,0,0)#(-0.1,-0.1) #[m] offset (x,y,theta) from center of robot
        self.move_viewcone(robot_pose)
        

        #Define simlated sensor parameters
        self.update_freq = 1 #[hz}
        self.detection_chance = 0.8 # P(detect|x) where x is in the viewcone

     
    def update(self,robot_pose,shape_layer,particle_filter,target_pose):
           self.move_viewcone(robot_pose)
           self.rescale_viewcone(robot_pose,shape_layer)
           return self.detect(particle_filter.particles, particle_filter.particle_probs,target_pose)   

    def move_viewcone(self,robot_pose):
        """Move the viewcone based on the robot's pose"""
        pose = (
                robot_pose[0] + self.offset[0],  
                robot_pose[1] + self.offset[1],
                robot_pose[2]
        )

        self.shape = self.viewcone
        transform = tuple(np.subtract(pose, self.view_pose))
        self.viewcone = self.move_shape(transform,rotation_pt=self.view_pose[0:2])
        self.view_pose = pose

    def rescale_viewcone(self,robot_pose,shape_layer):
        shape_layer = shape_layer.buffer(0)
        if self.shape.intersects(shape_layer):

            #calculate shadows for all shapes touching viewcone
            
            
            # origin = self.shape.project(map_object.shape)
            # shadow = affinity.scale(map_object.shape,xfact=1000,yfact=1000,origin=origin) #map portion invisible to the viewcone
            # self.shape = self.shape.difference(shadow)
            distance = Point(self.view_pose[0:2]).distance(shape_layer)
            scale = distance/self.max_view_dist * 1.3 #<>TODO: why the 1.3 multiplier?
            self.shape = affinity.scale(self.viewcone,xfact=scale,yfact=scale,origin=self.view_pose[0:2])
        else:
            self.shape = self.viewcone
                

    def sensor_model(self):
        """ """
        pass

    def detect(self,particles,particle_probs,target_pose):
        """ Update particles based on sensor model """
        #No target detection
        for i,particle in enumerate(particles):
            if self.shape.contains(Point(particle)):
                particle_probs[i] = particle_probs[i] * (1 - self.detection_chance)


        #renormalize
        particle_probs = particle_probs / sum(particle_probs)


        tar_point = Point(target_pose[0:2])
        if self.shape.contains(tar_point):
            min_distance = tar_point.distance(Point(particles[0]))
            for i,particle in enumerate(particles):
                dist = tar_point.distance(Point(particle))
                if dist < min_distance:
                    min_distance = dist
                    index_of_closest_particle = i
            particle_probs[:] = 0
            particle_probs[index_of_closest_particle] = 1



        return particle_probs


if __name__ == '__main__':

    #Pre-test config
    fig = plt.figure(1,figsize=(10,8)) 
    ax = fig.add_subplot(111)

    #Define Camera
    kinect = Camera()
    goal_points = [(0,0,0),(2,0,0),(2,0.5,90),(2,1.5,90),(2,1.7,90)]
    
    #Define Map and its objects
    bounds = (-5,-5,5,5)

    l = 1.2192 #[m] wall length
    w = 0.1524 #[m] wall width

    pose = (2.4,0,90)
    wall1 = MapObj('wall1',(l,w),pose)
    wall1.plot(plot_zones=False,alpha=0.8)
    # origin = wall1.shape.exterior.coords[1]
    # shadow = affinity.scale(wall1.shape,xfact=1000,yfact=1000,origin=origin) #map portion invisible to the viewcone
    # patch = PolygonPatch(shadow, facecolor=RED, alpha=0.5, zorder=2)
    # ax.add_patch(patch)

    pose = (2.5,2.5,45)
    wall2 = MapObj('wall2',(l,w),pose)
    wall2.plot(plot_zones=False,alpha=0.8)

    shape_layer = MultiPolygon((wall1.shape,wall2.shape))
    
    #Define Particle Filter
    target_pose = (10,10,0)
    particle_filter = ParticleFilter(bounds,"roy")
    particle_filter.update(kinect,target_pose)

    #Move camera and update the camera
    for point in goal_points:
        kinect.update(point,shape_layer,particle_filter,target_pose)
        kinect.plot(plot_zones=False,color=cnames['yellow'],alpha=0.4)  
    
    lim = 4
    ax.set_xlim([-lim/2,lim])
    ax.set_ylim([-lim/2,lim])

    particle_filter.plot()

    plt.show()
    