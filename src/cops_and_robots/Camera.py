from shapely.geometry import Polygon,Point

#sublcass of MapObj
class Camera(object):
	"""docstring for Camera"""
	def __init__(self, arg):
		super(Camera, self).__init__()
		self.arg = arg

		#Define nominal viewcone
		self.min_view_dist = 0.3 #[m]
		self.max_view_dist = 1.0 #[m]
		self.view_angle = 45 #[deg]
		self.viewcone = Polygon()
		

		self.update_freq = 1 #[hz}

		self.pose = ()

	def rescale_viewcone(self,map,pose):
		"""Rescale the viewcone to the maximum unencumbered distance"""
		pass

	def sensor_model(self):
		""" """
		pass

	def check_detection(self,target):
		""" Update camera once per update period """
		pass

if __name__ == '__main__':
    #Run a test to make sure camera's working
    pass