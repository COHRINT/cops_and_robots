import sys, time, logging, getch
from cops_and_robots.Cop import Cop

logger = logging.getLogger('moveTest')
logger.setLevel(logging.DEBUG)

cop = Cop('Deckard')

ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))

step = 100
self.speed = 0
self.radius = 0
x = 'a'

keymap = {  '.' : cop.faster,
			',' : cop.slower,
			'w' : cop.forward,
			's' : cop.backward,
			'a' : cop.left,
			'd' : cop.right,
			' ' : cop.stop }

tstep = 0.5

while x != 'z':
	x = getch.getch()
	logging.info('char:',x)

	cmd = cop.move()
	ser.write(cmd)
	time.sleep(tstep)

ser.close()
