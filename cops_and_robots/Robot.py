import math, numpy, logging, serial
from cops_and_robots.MapObj import MapObj

ser = serial.Serial('/dev/ttyUSB0',57600,timeout=1)


OPCODE = {
    # Getting Started
    'start': 128, 			#[128]
    'baud': 129,			#[129][Baud Code]
    'safe': 131,			#[130]
    #Demo Commdands
    'spot': 134,			#[128]
    'cover': 135,			#[135]
    'demo': 136,			#[136][Which-Demo]
    'cover-and-dock': 143, 	#[143]
    # Actuator Commands
    'drive': 137,			#[137][Vel-high-byte][Vel-low-byte][Rad-high-byte][Rad-low-byte]
    'drive-direct': 145,	#[145][Right-vel-high-byte][Right-vel-low-byte][Left-vel-high-byte][Left-vel-low-byte]
    'LEDs': 139,			#[139][LED][Color][Intensity]
    }


class Robot(MapObj):
	"""Base class for iRobot Create"""

	#Constants
	ROBOT_DIAMETER 	= 30 #[cm] <>TODO: VERIFY!
	RESOLUTION 		= 1000 	
	MAX_SPEED  		= 500 	#[mm/s]
	MAX_RADIUS 		= 2000 	#[mm]

	
	def __init__(self, name):
		"""Robot constructor

		:param name: string for the robot name
		"""		
		#Superclass attributes
		a 		 = numpy.linspace(0,2 * math.pi, RESOLUTION)
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
		self.speed 	 = 0
		self.radius  = 0

	def moveToTarget(self,target):
		"""Move directly to a target pose using A* for pathfinding

		:param target: desired pose
		"""
		pass
		#return result

	def move(self):
		"""Move based on robot's speed and radius
		"""
		
		#Translate speed to upper and lower bytes
		v = abs(self.speed) * (2**16 - 1)/ MAX_SPEED
		if self.speed > 0:
			v = "0x%04x" % v
		else:
			v = ((v ^ 0xffff) + 1) & 0xffff
			v = "0x%04x" % v
		v_h = int(v[2:4],16)
		v_l = int(v[4:6],16)

		#Translate radius to upper and lower bytes
		r = abs(self.radius) * (2**16 - 1)/ MAX_RADIUS
		if self.radius >= 0:
			r = "0x%04x" % r
		else:
			r = ((r ^ 0xffff) + 1) & 0xffff
			r = "0x%04x" % r
		r_h = int(r[2:4],16)
		r_l = int(r[4:6],16)


		#Generate serial drive command
		drive_params = [v_h,v_l,r_h,r_l]
		cmd = chr(OPCODE['drive'])
		print drive_params

		for i in drive_params:
			cmd = cmd + chr(i)
		result = cmd
		return result

	def randomTarget(self):
		"""Generate a random target pose on the map 

		:returns: pose (x,y,theta)
		"""
		pass
		#return result

	def checkBattery(self):
		"""Check the current battery status of the robot

		:returns: battery level from 0 to 1
		"""
		pass
		#return result

	def serialCommand(self,cmd):
		"""Send a serial command to the iRobot base

		:param cmd: character string accepted by the iRobot seraial interface
		"""
		pass

	def faster(self,step=10):
		"""Increase iRobot create speed

		:param: speed step size increase (default 10)
		"""
		logging.info('Faster!')
		if self.speed + step <= MAX_SPEED:
			self.speed = self.speed + step
		else:
			self.speed = MAX_SPEED

	def slower(self,step=10):
		"""Decrease iRobot create speed

		:param: speed step size increase (default 10)
		"""		
		logging.info('Slower...')
		if self.speed - step > 0:
			self.speed = self.speed - step
		else:
			self.speed = 0

	def forward(self):
		"""Move iRobot create forward at current speed
		"""
		logging.info('Forward!')
		self.radius = 0
		self.speed = self.speed		

	def backward(self):
		"""Move iRobot create forward at current speed
		"""
		logging.info('Backward!')
		self.radius = 0
		self.speed = -self.speed		

	def left(self,step=10):
		"""Turn left
		"""
		logging.info('Left!')
		if self.radius + step < MAX_RADIUS:
			self.radius = self.radius + step
		else:
			self.radius = MAX_RADIUS
		self.speed = self.speed	

	def right(self,step):
		"""Turn right
		"""
		logging.info('Right!')
		if self.radius - step > -MAX_RADIUS:
				self.radius = self.radius - step
		else:
			self.radius = -MAX_RADIUS
		self.speed = self.speed

	def stop(self):
		"""Stop the robot
		"""
		logging.info('STOP!')	
		self.speed = 0
