#!/usr/bin/env python
"""This module allows for the control of an iRobot Create base.

It is expected to be run on the iRobot's computer itself, where serial
commands are sent to the base. It is a superclass of Robot, which
defines any and all hardwre aspects of the robot control.

The 'base' is considered to be simply the iRobot Create's robotic
chassis. Any other reference to the iRobot Create nominally
includes the additional aspects of the robot (i.e. computer,
camera, actuators).

"""
__author__ = "Nick Sweet"
__copyright__ = "Copyright 2015, Cohrint"
__credits__ = ["Nick Sweet", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Nick Sweet"
__email__ = "nick.sweet@colorado.edu"
__status__ = "Development"

import math
import logging
import serial
import threading

# Make import Python 2 or Python 3 agnostic
try:
    import queue
except ImportError:
    import Queue as queue


class iRobotCreate(object):
    """Definition and implementation of iRobot Create functionality.

    Largely controls movement of the robot, but also provides
    interfaces for the robot's bump and cliff sensors.

    Attributes
    ----------
    DIAMETER : float
        Diameter of the circular robot in meters.
    MAX_SPEED : float
        Maximum linear speed in mm/s.
    MAX_TURN_RADIUS : float
        Maximum radius of the turning circle in mm.
    RAD_STRAIGHT : float
        Special radius for linear movement.
    RAD_CW : float
        Special radius of a turning circle for in-place clockwise rotation.
    RAD_CCW : float
        Special radius of a turning circle for in-place counterclockwise
        rotation.
    OPCODE : dict
        All opcodes required to communicate with the iRobot Create.
    SENSOR_PKT : dict
        Identifiers for each sensor packet to be received from the iRobot
        Create.
    CHARGE_MODE : dict
        Identifiers for each charging mode.
    OI_MODE : dict
        Identifiers for each operation mode.

    """
    # Class Constants
    DIAMETER = 0.34  # [m] (appoximate)
    MAX_SPEED = 500  # [mm/s] linear speed
    MAX_TURN_RADIUS = 2000  # [mm] roation radius

    # Special movement radii
    RAD_STRAIGHT = 2 ** 15 - 1
    RAD_CW = 2 ** 16 - 1
    RAD_CCW = 1

    OPCODE = {
        # Getting Started
        'start': 128,  # [128]
        'baud': 129,  # [129][Baud Code]
        'safe': 131,  # [130]
        'full': 132,  # [132]
        # Demo Commands
        'spot': 134,  # [128]
        'cover': 135,  # [135]
        'demo': 136,  # [136][Which-Demo]
        'cover-and-dock': 143,  # [143]
        # Actuator Commands
        'drive': 137,  # [137][Vel-high-byte][Vel-low-byte][Rad-high-byte]
                       # [Rad-low-byte]
        'drive-direct': 145,  # [145][Right-vel-high-byte][Right-vel-low-byte]
                              # [Left-vel-high-byte][Left-vel-low-byte]
        'LEDs': 139,  # [139][LED][Color][Intensity]
        # Sensor Commands
        'sensors': 142,  # [142][Packet Id]
        'stream': 148,  # [148][Num-packets][Pack-id-1][Pack-id-2]...
        'query-list': 149,  # [149][Pack-id-1][Pack-id-2]...
        'stream_toggle': 150  # [150][0 or 1] to [stop or resume] the stream
    }

    SENSOR_PKT = {
        'bump-wheel-drop': 7,  # [4] caster; [3] wheel left; [2] wheel right;
                               # [1] bump left; [0] bump right
        'wall': 8,  # [0]
        'cliff-left': 9,  # [0]
        'cliff-front-left': 10,  # [0]
        'cliff-front-right': 11,  # [0]
        'cliff-right': 12,  # [0]
        'virtual-wall': 13,  # [0]
        'overcurrent': 14,  # [4] left wheel; [3] right wheel; [2] LDO2;
                            # [1] LDO0 [0] LDO1
        'IR': 17,  # [7-0] see Create Open Interface
        'buttons': 18,  # [2] Advance; [0] Play
        'distance': 19,  # [15-0] mm since last requested
        'angle': 20,  # [15-0] degrees since last requested
        'charging': 21,  # [5] fault; [4] waiting; [3] trickle; [2] full;
                         # [1] reconditioning; [0] none
        'voltage': 22,  # [15-0] battery voltage in mV
        'current': 23,  # [15-0] battery current in mA (+ for charging)
        'temperature': 24,  # [7-0] battery temperature in degrees Celsius
        'charge': 25,  # [15-0] battery charge in mAh
        'capacity': 26,  # [15-0] battery capacity in mAh
        'wall-signal': 27,  # [11-0] sensor signal strength
        'cliff-left-signal': 28,  # [11-0] sensor signal strength
        'cliff-front-left-signal': 29,  # [11-0] sensor signal strength
        'cliff-front-right-signal': 30,  # [11-0] sensor signal strength
        'cliff-right-signal': 31,  # [11-0] sensor signal strength
        'OI-mode': 35  # [0] off; [1] passive; [2] safe; [3] full
    }

    CHARGE_MODE = {
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

    def __init__(self):
        self.charge_mode = iRobotCreate.CHARGE_MODE['none']
        self.battery_charge = 0
        self.battery_capacity = 0
        self.bump_left = False
        self.bump_right = False
        self.speed = 0
        self.rotation_radius = iRobotCreate.MAX_TURN_RADIUS
        self.oi_mode = iRobotCreate.OI_MODE['off']
        self.cmd_queue = queue.Queue()

        # Add logger
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

    def start_base_cx(self):
        """Start the thread to communicate with the iRobot Create base."""
        self.base_t = threading.Thread(target=self._base_thread_f)
        self.base_t_stop = threading.Event()  # Allow graceful killing
        self.base_t.start()

    def stop_base_cx(self):
        """Stop the thread communicating with the iRobot Create base."""
        # allowing ctrl-c to close thread (see
        # http://www.regexprn.com/2010/05/killing-multithreaded-python-programs.html)
        while True:
            try:
                self.base_t.join(1)
            except (KeyboardInterrupt, SystemExit):
                self.base_t_stop.set()
                break

    def _base_thread_f(self):
        """Thread function to implement serial communication with the base.
        """
        # Connect to the serial port
        portstr = '/dev/ttyUSB0'
        try:
            ser = serial.Serial(portstr, 57600, timeout=1)
        except Exception:
            ser = "fail"
            logging.error("Failed to connect to {}".format(portstr))
            return ser

        # Enable commanding of the robot
        ser.write(chr(iRobotCreate.OPCODE['start'])
                  + chr(iRobotCreate.OPCODE['full']))

        # Loop to poll sensors and command the base
        while not self.base_t_stop.is_set():
            # Define sensors to be polled
            num_packets = 5
            expected_response_length = 7  # Number of data bytes returned
            sensors = [iRobotCreate.SENSOR_PKT['OI-mode'],
                       iRobotCreate.SENSOR_PKT['charging'],
                       iRobotCreate.SENSOR_PKT['charge'],
                       iRobotCreate.SENSOR_PKT['capacity'],
                       iRobotCreate.SENSOR_PKT['bump-wheel-drop']
                       ]

            # Create the packet to be transmitted to the base
            tx_packet = chr(iRobotCreate.OPCODE['query-list'])\
                + chr(num_packets)
            for sensor in sensors:
                tx_packet = tx_packet + chr(sensor)

            # Flush the stream and request data packets
            ser.flushInput()
            ser.write(tx_packet)
            logging.debug("Transmitted packet: {}".format(tx_packet))

            # Read the response
            try:
                response = ser.read(size=expected_response_length)
                logging_resp = [(ord(x)) for x in response]
                logging.debug("Received packet: {}".format(logging_resp))
            except:
                response = ''
                logging.error("Failed to read from {}".format(portstr))

            # Evaluate the response
            if len(response) < expected_response_length:
                logging.error("Unexpected response length ({} instead of {})"
                              .format(len(response), expected_response_length))
            else:
                # Break up returned bytes
                oi_mode_byte = ord(response[0])
                charge_mode_byte = ord(response[1])
                charge_bytes = [ord(response[2]), ord(response[3])]
                capacity_bytes = [ord(response[4]), ord(response[5])]
                bump_byte = ord(response[6])

                # Update OI mode
                if oi_mode_byte & 8:
                    self.oi_mode = iRobotCreate.OI_MODE['full']
                elif oi_mode_byte & 4:
                    self.oi_mode = iRobotCreate.OI_MODE['safe']
                elif oi_mode_byte & 2:
                    self.oi_mode = iRobotCreate.OI_MODE['passive']
                elif not oi_mode_byte == 1:
                    self.oi_mode = iRobotCreate.OI_MODE['off']
                else:
                    logging.error('Incorrect OI mode returned!')

                # Update charging mode
                if charge_mode_byte & 32:
                    self.charge_mode = iRobotCreate.CHARGE_MODE['fault']
                elif charge_mode_byte & 16:
                    self.charge_mode = iRobotCreate.CHARGE_MODE['waiting']
                elif charge_mode_byte & 8:
                    self.charge_mode = iRobotCreate.CHARGE_MODE['trickle']
                elif charge_mode_byte & 4:
                    self.charge_mode = iRobotCreate.CHARGE_MODE['full']
                elif charge_mode_byte & 2:
                    self.charge_mode = iRobotCreate \
                        .CHARGE_MODE['reconditioning']
                elif not charge_mode_byte == 1:
                    self.charge_mode = iRobotCreate.CHARGE_MODE['none']
                else:
                    logging.error('Incorrect charging mode returned!')

                # Update battery characteristics
                self.battery_capacity = capacity_bytes[0] * 256 \
                    + capacity_bytes[1]
                self.battery_charge = charge_bytes[0] * 256 + charge_bytes[1]
                if self.battery_charge > self.battery_capacity:
                    self.battery_charge = 0
                    logging.warn("Charge reported greater than capacity.")
                logging.debug("Capacity: {} \t Charge: {}"
                              .format(self.battery_capacity,
                                      self.battery_charge))

                # Update bump sensor readings
                self.bump_right = bump_byte & 1
                self.bump_left = bump_byte & 2

            # Loop through messages in the command queue
            while not self.cmd_queue.empty():
                cmd = self.cmd_queue.get()
                ser.write(cmd)

        # Stop the stream
        ser.write(chr(iRobotCreate.OPCODE['stream_toggle']) + chr(0))
        logging.debug("Exiting gracefully!")
        ser.close()

    def _int2ascii(self, integer):
        """Take a 16-bit signed integer and convert it to two ascii characters.

        Parameters
        ----------
        integer : int
            Integer value no larger than (+/-)2^15.

        Returns
        -------
        tuple
            High and low byte ascii characters.
        """

        if integer < 0:
            low_byte = abs(abs(integer) | ~pow(2, 8) + 1)
            high_byte = abs(abs(integer >> 8) | ~pow(2, 8) + 1)
        else:
            low_byte = integer & (pow(2, 8) - 1)
            high_byte = integer >> 8 & (pow(2, 8) - 1)

        low_char = chr(low_byte)
        high_char = chr(high_byte)

        return (high_char, low_char)

    def move(self):
        """Move based on robot's speed and radius.

        Returns
        -------
        str
            A string of drive commands as hex characters.
        """

        # Translate speed to upper and lower bytes
        (s_h, s_l) = self._int2ascii(self.speed)

        # Translate radius to upper and lower bytes
        (r_h, r_l) = self._int2ascii(self.rotation_radius)

        # Generate serial drive command
        drive_params = [s_h, s_l, r_h, r_l]
        drive_params.insert(0, chr(iRobotCreate.OPCODE['drive']))
        logging.info(drive_params)

        result = ''.join(drive_params)  # Convert to a str
        return result

    def faster(self, step=100):
        """Increase iRobot create speed.

        :param step: speed step size increase.
        :type step: int.
        """
        logging.info('Faster!')
        if self.speed + step <= iRobotCreate.MAX_SPEED:
            self.speed = self.speed + step
        else:
            self.speed = iRobotCreate.MAX_SPEED

    def slower(self, step=100):
        """Decrease iRobot create speed.

        :param step: speed step size increase.
        :type step: int.
        """
        logging.info('Slower...')
        if self.speed - step > 0:
            self.speed = self.speed - step
        else:
            self.speed = 0

    def forward(self):
        """Move iRobot create forward at current speed.
        """
        logging.info('Forward!')
        self.rotation_radius = iRobotCreate.RAD_STRAIGHT
        self.speed = abs(self.speed)

    def backward(self):
        """Move iRobot create forward at current speed.
        """
        logging.info('Backward!')
        self.rotation_radius = iRobotCreate.RAD_STRAIGHT
        self.speed = -abs(self.speed)

    def turn(self, radius):
        """Move in a circle.

        Parameters
        ----------
        radius : float
            Longer radii make Create drive straighter, while shorter radii
            make the Create turn more. The radius is measured from the center
            of the turning circle to the center of Create.

            A Drive command with a positive velocity and a positive radius
            makes Create drive forward while turning toward the left. A
            negative radius makes Create turn toward the right.

        """
        logging.info('Turning!')
        if abs(radius) > iRobotCreate.MAX_TURN_RADIUS:
                radius = int(math.copysign(iRobotCreate.MAX_TURN_RADIUS,
                             radius))
        self.rotation_radius = radius

    def rotate_ccw(self):
        """Rotate in place counterclockwise.
        """
        logging.info('Rotate Left!')
        self.rotation_radius = iRobotCreate.RAD_CCW

    def rotate_cw(self):
        """Rotate in place clockwise.
        """
        logging.info('RotateRight!')
        self.rotation_radius = iRobotCreate.RAD_CW

    def stop(self):
        """Stop the iRobotCreate base.
        """
        logging.info('STOP!')
        self.speed = 0
