import math, numpy, logging, serial
from cops_and_robots.MapObj import MapObj
from cops_and_robots.Map import Map

class Robot(MapObj):
    """Base class for iRobot Create"""

    #Constants
    DIAMETER        = 30 #[cm] <>TODO: VERIFY!
    RESOLUTION      = 1000  
    MAX_SPEED       = 500   #[mm/s]
    MAX_RADIUS      = 2000  #[mm]

    #Special movement radii
    RAD_STRAIGHT    = 2**15 - 1
    RAD_CW          = 2**16 - 1
    RAD_CCW         = 1

    OPCODE = {
    # Getting Started
    'start': 128,           #[128]
    'baud': 129,            #[129][Baud Code]
    'safe': 131,            #[130]
    'full': 132,            #[132]
    #Demo Commdands
    'spot': 134,            #[128]
    'cover': 135,           #[135]
    'demo': 136,            #[136][Which-Demo]
    'cover-and-dock': 143,  #[143]
    # Actuator Commands
    'drive': 137,           #[137][Vel-high-byte][Vel-low-byte][Rad-high-byte][Rad-low-byte]
    'drive-direct': 145,    #[145][Right-vel-high-byte][Right-vel-low-byte][Left-vel-high-byte][Left-vel-low-byte]
    'LEDs': 139,            #[139][LED][Color][Intensity]
    }

    #Add logger
    logger = logging.getLogger('moveTest')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    #Connect to the serial port
    portstr = '/dev/ttyUSB0'
    try:
        ser = serial.Serial(portstr,57600,timeout=1)
    except Exception, e:
        ser = "fail"
        logging.error("Failed to connect to %s" % portstr)
        # raise e
    
    
    def __init__(self, name):
        """Robot constructor

        :param name: string for the robot name
        """     
        #Superclass attributes
        a        = numpy.linspace(0,2 * math.pi, Robot.RESOLUTION)
        circ_x   = [(Robot.DIAMETER / 2 * math.sin(b)) for b in a]
        circ_y   = [(Robot.DIAMETER / 2 * math.cos(b)) for b in a]
        shape    = zip(circ_x,circ_y)           #draw a circle with radius ROBOT_DIAMETER/2 around centroid
        centroid = {'x':0,'y':0,'theta':0}      #Start at origin
        pose     = {'x':0,'y':0,'theta':0}      #Start at origin
        MapObj.__init__(self,name,shape,centroid,pose)

        #Class attributes
        self.target  = {'x':0,'y':0,'theta':0}  #Start at origin
        self.battery = 0
        self.bump    = False
        self.map     = Map()
        self.speed   = 0
        self.radius  = Robot.MAX_RADIUS

    def moveToTarget(self,target):
        """Move directly to a target pose using A* for pathfinding

        :param target: desired pose
        """
        pass
        #return result

    def int2ascii(self,integer):
        """Takes a 16-bit signed integer and converts it to two ascii characters

        :param integer: integer value no larger than (+/-)2^15
        :returns: high and low ascii characters
        """
        
        if integer < 0:
            low_byte    = abs(abs(integer) | ~pow(2,8)+1)
            high_byte   = abs(abs(integer>>8) | ~pow(2,8)+1)
        else:
            low_byte    = integer & (pow(2,8)-1)
            high_byte   = integer>>8 & (pow(2,8)-1)            

        low_char    = chr(low_byte)
        high_char   = chr(high_byte)

        return (high_char, low_char)

    def move(self):
        """Move based on robot's speed and radius

        :returns: string of hex characters
        """
        
        #Translate speed to upper and lower bytes
        (s_h, s_l) = self.int2ascii(self.speed)

        #Translate radius to upper and lower bytes
        (r_h, r_l) = self.int2ascii(self.radius)

        #Generate serial drive command
        drive_params = [s_h,s_l,r_h,r_l]
        drive_params.insert(0,chr(Robot.OPCODE['drive']))
        logging.info(drive_params)

        result = ''.join(drive_params) #Convert to a str
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

        :param cmd: character string accepted by the iRobot serial interface
        """
        pass

    def faster(self,step=100):
        """Increase iRobot create speed

        :param: speed step size increase (default 100)
        """
        print 'test'
        logging.info('Faster!')
        if self.speed + step <= Robot.MAX_SPEED:
            self.speed = self.speed + step
        else:
            self.speed = Robot.MAX_SPEED

    def slower(self,step=100):
        """Decrease iRobot create speed

        :param: speed step size increase (default 100)
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
        self.radius = Robot.RAD_STRAIGHT
        self.speed = self.speed     

    def backward(self):
        """Move iRobot create forward at current speed
        """
        logging.info('Backward!')
        self.radius = Robot.RAD_STRAIGHT
        self.speed = -self.speed        

    def turn(self,radius):
        """Move in a circle

        :param radius: The longer radii make Create drive straighter, while the shorter radii make Create turn more. The radius is measured from the center of the turning circle to the center of Create. A Drive command with a positive velocity and a positive radius makes Create drive forward while turning toward the left. A negative radius makes Create turn toward the right.
        """
        logging.info('Turning!')
        if abs(radius) > Robot.MAX_RADIUS:
                radius  = int(math.copysign(Robot.MAX_RADIUS,radius))
        self.radius     = radius
        self.speed      = self.speed

    def rotateCCW(self):
        """Rotate in place counterclockwise
        """
        logging.info('Rotate Left!')
        self.radius = Robot.RAD_CCW
        self.speed = self.speed 

    def rotateCW(self):
        """Rotate in place clockwise
        """
        logging.info('RotateRight!')
        self.radius = Robot.RAD_CW
        self.speed = self.speed

    def stop(self):
        """Stop the robot
        """
        logging.info('STOP!')   
        self.speed = 0
