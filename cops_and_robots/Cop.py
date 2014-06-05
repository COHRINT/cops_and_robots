from cops_and_robots.Robot import Robot

class Cop(Robot):
	"""docstring for Cop"""
	def __init__(self, name = "Deckard"):
		#Superclass attributes
		Robot.__init__(self,name)
		
		