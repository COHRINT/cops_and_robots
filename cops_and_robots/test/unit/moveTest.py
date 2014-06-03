import sys, time, logging
import getch, Robot

logger = logging.getlogger('moveTest')
logger.setLevel(logging.DEBUG)

robot = Robot('Deckard')

ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))

step = 100
self.speed = 0
self.radius = 0
x = 'a'

keymap = {  '.' : robot.faster,
			',' : robot.slower,
			'w' : robot.forward,
			's' : robot.backward,
			'a' : robot.left,
			'd' : robot.right,
			' ' : robot.stop }


while x != 'z':
	x = getch()
	logging.info('char:',x)
	ser.write(robot.move(self.speed,self.radius))
	time.sleep(1)

ser.close()
