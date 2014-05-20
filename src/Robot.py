import math, numpy, MapObj

class Robot(MapObj):
	"""docstring for Robot"""

	#Constants
	ROBOT_DIAMETER = 30 #[cm] <>TODO: VERIFY!
	RESOLUTION = 1000
	
	def __init__(self, name):
		#Superclass attributes
		a = numpy.linspace(0,2*math.pi,RESOLUTION)
		circ_x 	 = [(ROBOT_DIAMETER / 2 * math.sin(b)) for b in a]
		circ_y 	 = [(ROBOT_DIAMETER / 2 * math.cos(b)) for b in a]
		shape 	 = zip(circ_x,circ_y) 			#draw a circle with radius ROBOT_DIAMETER/2 around centroid
		centroid = {'x':0,'y':0,'theta':0} 		#Start at origin
		pose 	 = {'x':0,'y':0,'theta':0}		#Start at origin
		MapObj.__init__(self,name,shape,centroid,pose)

		#Class attributes
		self.target  = {'x':0,'y':0,'theta':0}	#Start at origin
		self.battery = 0
		self.bump 	 = False
		self.map 	 = Map()

	def move(self,target):
		pass
		#return result

	def randomTarget(self):
		pass
		#return result

	def checkBattery(self):
		pass
		#return result

	def serialCommand(self,cmd):
		pass
