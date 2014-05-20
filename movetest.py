#!/usr/bin/env python

import serial
import time
import create_driver
import getch

getch = getch._Getch()
ser = serial.Serial('/dev/ttyUSB0',57600,timeout=1)

MAX_SPEED = 500 	#[mm/s]
MAX_RADIUS = 2000 	#[mm]

OPCODE = {
    # Getting Started
    'start': 128, 			#[128]
    'baud': 129,			#[129][Baud Code]
    'safe': 131,			#[130]
    'full': 132,			#[131]
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


def drive_command(speed,radius):
	#speed: -500 to 500 mm/s
	#radius: -MAX_RADIUS to MAX_RADIUS mm (small means more turning)

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
	return cmd


#Turn in place clockwise = 0xFFFF 
#Turn in place counter-clockwise = 0x0001

#Use commands
n = 0
while n < 3 :
    ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))
    ser.write(drive_command(100,0))
    time.sleep(1)
    ser.write(drive_command(-100,0))
    time.sleep(1)
    n = n + 1
ser.write(drive_command(0,0))


step = 100
speed = 0
radius = 0
x = 'a'

# ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))

while x != 'z':

	x = getch()
	print 'char:',x

	if x == '.':
		print 'Faster!'
		if speed < MAX_SPEED:
			speed = speed + step
		else:
			speed = MAX_SPEED
	if x == ',':
		if speed > 0:
			speed = speed - step
			print 'Slower...'
		else:
			speed = 0
	
	if x == 'w':
		radius = 0
		speed = speed
		print 'Forward!'
	elif x == 'a':
		if radius < MAX_RADIUS:
			radius = radius + step
		else:
			radius = MAX_RADIUS
		speed = speed	
		print 'Left!'
	elif x == 's':
		radius = 0
		speed = -speed
		print 'Backward!'
	elif x == 'd':
		if radius > -MAX_RADIUS:
			radius = radius - step
		else:
			radius = -MAX_RADIUS
		speed = speed
		print 'Right!'
	# else:
	# 	speed = 0
	# 	radius = 0
	# 	print 'Stop'
	
	if x == ' ':
		speed = 0
		radius = 0
		print 'STAHP'

	ser.write(drive_command(speed,radius))
	time.sleep(1)

ser.close()

