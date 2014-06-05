import sys, time, logging, getch
from cops_and_robots.Cop import Cop

logger = logging.getLogger('moveTest')
logger.addHandler(logging.StreamHandler()) #output to console
logging.basicConfig(level=logging.DEBUG)

cop = Cop('Deckard')
OPCODE = cop.OPCODE
ser = cop.ser

ser.write(chr(OPCODE['start']) + chr(OPCODE['full']))

step = 100
cop.speed = 0
cop.radius = cop.MAX_RADIUS
x = 'a'

keymap = {  '.' : lambda: cop.faster(step),
            ',' : lambda: cop.slower(step),
            'w' : lambda: cop.forward(),
            's' : lambda: cop.backward(),
            'a' : lambda: cop.left(step*10),
            'd' : lambda: cop.right(step*10),
            ' ' : lambda: cop.stop() }

tstep = 0.5

while x != 'z':
    x = getch.getch()

    try:
        keymap[x]()
    except Exception, e:
        logging.error('%s is not a viable command',x)
    
    
    logging.info('char: %s',x)
    logging.info('speed: %d mm/s',cop.speed)
    logging.info('radius: %d mm/s',cop.radius)

    cmd = cop.move()
    ser.write(cmd)
    time.sleep(tstep)

ser.close()
