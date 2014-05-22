import math, numpy, MapObj
import serial, create_driver

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

	def moveToTarget(self,target):
		pass
		#return result

	def move(self,speed,radius):
		#Constants
		MAX_SPEED  = 500 	#[mm/s]
		MAX_RADIUS = 2000 	#[mm]

		#Saturate Input
		if speed > MAX_SPEED:
			speed = MAX_SPEED
		elif speed < -MAX_SPEED:
			speed = -MAX_SPEED
		if radius > MAX_RADIUS:
			radius = MAX_RADIUS
		elif radius < -MAX_RADIUS:
			radius = -MAX_RADIUS
		
		#Translate speed to upper and lower bytes
		v = abs(speed) * (2**16 - 1)/ MAX_SPEED
		if speed > 0:
			v = "0x%04x" % v
		else:
			v = ((v ^ 0xffff) + 1) & 0xffff
			v = "0x%04x" % v
		v_h = int(v[2:4],16)
		v_l = int(v[4:6],16)

		#Translate radius to upper and lower bytes
		r = abs(radius) * (2**16 - 1)/ MAX_RADIUS
		if radius >= 0:
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
		pass
		#return result

	def checkBattery(self):
		pass
		#return result

	def serialCommand(self,cmd):
		pass
