import Robot
import getch, time

deckard = Robot('Deckard')

ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))

step = 100
speed = 0
radius = 0
x = 'a'

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

	ser.write(deckard.move(speed,radius))
	time.sleep(1)

ser.close()
