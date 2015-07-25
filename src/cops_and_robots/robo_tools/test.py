class Test(object):
        
    def __init__(self):
        self.x=1
        self.y=2
        self.theta=0.6
        self.pose=[self.x,self.y,self.theta]
        self.pose[0]=7
        print self.x
        self.y=0.1
        print self.pose

if __name__ == '__main__':
    Test()
    str_1="Hey"
    str_2="Hey"
    str_3="Joe"
    print str_1==str_2
    print str_1 is str_2
    print str_1==str_3
    print str_1 is str_3