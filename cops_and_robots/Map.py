class Map(object):
	"""docstring for Map"""

	BOUNDS_1M  = [(0,0),(0,100),(100,100),(100,0)] #[cm] one meter square
	BOUNDS_10M = [[y * 10 for y in x] for x in BOUNDS_1M] #[cm] 10 meter square


	def __init__(self):
		self.bounds = Map.BOUNDS_10M
		self.walls = []

	def addWall(self,wall):
		"""Append a ``wall`` to the Map

		:param wall: Wall object
		"""
		self.walls.append(wall)

		