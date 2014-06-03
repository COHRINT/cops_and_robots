from cops_and_robots import Robot

class Cop(Robot):
	"""docstring for Cop"""
	def __init__(self, arg):
		#Superclass attributes
		name = "Deckard"
		Robot.__init__(self,name)
		
		