import math, numpy, logging, serial, socket, threading
from cops_and_robots.MapObj import MapObj
from cops_and_robots.Map import Map

class Robot(MapObj):
    """Class for controlling iRobot Create. Will generate a 'base' thread to maintain
    serial communication with the iRobot base."""

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
    # Demo Commands
    'spot': 134,            #[128]
    'cover': 135,           #[135]
    'demo': 136,            #[136][Which-Demo]
    'cover-and-dock': 143,  #[143]
    # Actuator Commands
    'drive': 137,           #[137][Vel-high-byte][Vel-low-byte][Rad-high-byte][Rad-low-byte]
    'drive-direct': 145,    #[145][Right-vel-high-byte][Right-vel-low-byte][Left-vel-high-byte][Left-vel-low-byte]
    'LEDs': 139,            #[139][LED][Color][Intensity]
    # Sensor Commands
    'stream': 148,          #[148][Num-packets][Pack-id-1][Pack-id-2]...
    }

    SENSOR_PKT = {
    'bump-wheel-drop': 7,   #[4] caster; [3] wheel left; [2] wheel right; [1] bump left; [0] bump right
    'wall': 8,              #[0]
    'cliff-left': 9,        #[0]
    'cliff-front-left': 10, #[0]
    'cliff-front-right': 11,#[0]
    'cliff-right': 12,      #[0]
    'virtual-wall': 13,     #[0]
    'overcurrent': 14,      #[4] left wheel; [3] right wheel; [2] LDO2; [1] LDO0 [0] LDO1
    'IR': 17,               #[7-0] see Create Open Interface
    'buttons': 18,          #[2] Advance; [0] Play
    'distance': 19,         #[15-0] mm since last requested
    'angle': 20,            #[15-0] degrees since last requested
    'charging': 21,         #[5] fault; [4] waiting; [3] trickle; [2] full; [1] reconditioning; [0] none        
    'voltage': 22,          #[15-0] battery voltage in mV
    'current': 23,          #[15-0] battery current in mA (+ for charging)
    'temperature': 24,      #[7-0] battery temperature in degrees Celsius
    'charge':25,            #[15-0] battery charge in mAh
    'capacity':26,          #[15-0] battery capacity in mAh
    'wall-signal':27,       #[11-0] sensor signal strength
    'cliff-left-signal':28, #[11-0] sensor signal strength
    'cliff-front-left-signal':29, #[11-0] sensor signal strength
    'cliff-front-right-signal':30,#[11-0] sensor signal strength
    'cliff-right-signal':31,#[11-0] sensor signal strength
    'OI-mode':35            #[0] off; [1] passive; [2] safe; [3] full
    }

    CHARGING_MODE = {
    'none': 0,
    'reconditioning': 1,
    'full': 2,
    'trickle': 3,
    'waiting': 4,
    'fault': 5
    }

    OI_MODE = {
    'off': 0,
    'passive': 1,
    'safe': 2,
    'full': 3
    }

    #Add logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
        
    def __init__(self):
        """Robot constructor
        """     
        #Superclass attributes
        a        = numpy.linspace(0,2 * math.pi, Robot.RESOLUTION)
        circ_x   = [(Robot.DIAMETER / 2 * math.sin(b)) for b in a]
        circ_y   = [(Robot.DIAMETER / 2 * math.cos(b)) for b in a]
        shape    = zip(circ_x,circ_y)           #draw a circle with radius ROBOT_DIAMETER/2 around centroid
        centroid = {'x':0,'y':0,'theta':0}      #Start at origin
        pose     = {'x':0,'y':0,'theta':0}      #Start at origin
        name     = socket.gethostname()
        MapObj.__init__(self,name,shape,centroid,pose)

        #Class attributes
        self.target         = {'x':0,'y':0,'theta':0}  #Start at origin
        self.charging_mode  = Robot.CHARGING_MODE['none']
        self.battery_charge = 0
        self.battery_capacity = 0
        self.bump_left      = False
        self.bump_right     = False
        self.map            = Map('fleming_no_walls')
        self.speed          = 0
        self.radius         = Robot.MAX_RADIUS
        self.OI_mode        = Robot.OI_MODE['off'] 

        #Spawn base thread
        t = threading.Thread(target=self.base)
        t.start()
        self._stop = threading.Event()


    def base(self):
        """Seperate thread taking care of serial communication with the iRobot base
        """
        #Connect to the serial port
        portstr = '/dev/ttyUSB0'
        try:
            ser = serial.Serial(portstr,57600,timeout=1)
        except Exception, e:
            ser = "fail"
            logging.error("Failed to connect to {}".format(portstr))
            return ser
            # raise e

        #Enable commanding of the robot
        ser.write(chr(Robot.OPCODE['start']) + chr(Robot.OPCODE['full']))
        #ser.write(chr(OPCODE['start']) + chr(OPCODE['safe']))

        while(True):
            logging.debug("Entered sensor stream loop")

            #Start the sensor stream from the iRobot create
            num_packets = 5
            expected_response_length = 15 
            sensors = [ Robot.SENSOR_PKT['OI-mode'], 
                        Robot.SENSOR_PKT['charging'], 
                        Robot.SENSOR_PKT['charge'],
                        Robot.SENSOR_PKT['capacity'],
                        Robot.SENSOR_PKT['bump-wheel-drop'] ]

            TX_packet = chr(Robot.OPCODE['stream']) + chr(num_packets)
            for sensor in sensors:
                TX_packet = TX_packet + chr(sensor)
                          
            ser.write(TX_packet)
            logging.debug("Transmitted packet: {}".format(TX_packet))
            
            try:
                response = ser.read(expected_response_length)
                logging.debug("Received packet: {}".format(response))
            except Exception, e:
                response = ''
                logging.error("Failed to read from {}".format(portstr))


            if len(response) < expected_response_length:
                logging.error("Unexpected response length ({} instead of {})".format(len(response),expected_response_length) )

            #Break up returned bytes
            OI_mode_byte    = ord(response[3])
            charging_byte   = ord(response[5])
            charge_bytes    = [ord(response[7]),ord(response[8])]
            capacity_bytes  = [ord(response[10]),ord(response[11])]
            bump_byte       = ord(response[13])

            #Update OI mode
            if OI_mode_byte & 1:
                self.OI_mode = Robot.OI_MODE['off']
            elif OI_mode_byte & 2:
                self.OI_mode = Robot.OI_MODE['passive']
            elif OI_mode_byte & 4:
                self.OI_mode = Robot.OI_MODE['safe']
            elif OI_mode_byte & 8:
                self.OI_mode = Robot.OI_MODE['full']
            else:
                logging.error('Incorrect OI mode returned!')

            #Update charging mode
            if charging_byte & 1:
                self.charging_mode = Robot.CHARGING_MODE['none']
            elif charging_byte & 2:
                self.charging_mode = Robot.CHARGING_MODE['reconditioning']
            elif charging_byte & 2:
                self.charging_mode = Robot.CHARGING_MODE['full']
            elif charging_byte & 2:
                self.charging_mode = Robot.CHARGING_MODE['trickle']
            elif charging_byte & 2:
                self.charging_mode = Robot.CHARGING_MODE['waiting']
            elif charging_byte & 2:
                self.charging_mode = Robot.CHARGING_MODE['fault']
            else:
                logging.error('Incorrect Charging mode returned!')                                                                

            #Update battery characteristics
            self.battery_capacity = capacity_bytes[0]*256 + capacity_bytes[1]
            self.battery_charge   = charge_bytes[0]*256 + charge_bytes[1]
            logging.debug("Capacity: {} \n Charge: {}".format(self.battery_capacity, self.battery_charge))

            #Update bump sensor readings
            self.bump_right = bump_byte & 1
            self.bump_left  = bump_byte & 2

            #Start the sensor stream from the Vicon system
            #self.pose

            #Loop through messages in the command queue

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
        self.speed = abs(self.speed)     

    def backward(self):
        """Move iRobot create forward at current speed
        """
        logging.info('Backward!')
        self.radius = Robot.RAD_STRAIGHT
        self.speed = -abs(self.speed)        

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
