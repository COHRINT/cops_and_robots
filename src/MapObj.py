class MapObj(object):

    def __init__(self,name,shape,centroid,pose):
        self.name       = name      #sting identifier
        self.shape      = shape     #nx2 array containing all (x,y) vertices for the object shape
        self.centroid   = centroid  #(x,y,theta) coordinates of the centroid in the object frame
        self.pose       = pose      #(x,y,theta) coordinates of the centroid in the global frame

    def __str___(self):
        return "%s is located at (%d,%d), pointing at %d" % (self.name, self.centroid['x'],self.centroid['y'],self.centroid['theta'])

    def getName(self):
        return self.name

    def getShape(self):
        return self.shape

    def getCentroid(self):
        return self.centroid

    def getPose(self):
        return self.pose

    def setName(self, name):
        self.name = name

    def setShape(self, shape):
        self.shape = shape

    def setCentroid(self, centroid):
        self.centroid = centroid

    def setPose(self, pose):
        self.pose = pose 
