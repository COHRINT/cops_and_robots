##Cops and Robots

Cops and Robots is a testbed for experimental research done by the COHRINT lab at the University of Colorado, Boulder. This repository is a code backend to support robot control, human interface and dynamic target tracking.

The testbed is broken up into three parts: python simulation, [gazebo simulation](http://gazebosim.org/), and hardware experiment. The code for the python simulation lives in `src/cops_and_robots`, the code for the gazebo simulation lives at the root level of this project, and the code for the experiment is not yet ready.

**Note**: all code, including on the Master branch, is volatile. Use at your own risk.

See [the documentation](http://recuv.colorado.edu/~sweet/cops_and_robots) for a low-level discussion of the Python code.

## Installation Instructions
*For Unix-based systems only. Sorry, Windows users, but you're on your own.*

Create and load a [virtual environment](https://virtualenv.pypa.io/en/latest/):
```
pip install virtualenv
mkdir ~/virtual_environments # or whatever you'd like to name your folder
cd ~/virtual_environments 
virtualenv cops_and_robots # or whatever you'd like to name your environment
source cops_and_robots/bin/activate
```

Install [SciPy](http://www.scipy.org/):
```
# On Linux only:
sudo apt-get install libatlas-base-dev gfortran

# On OS X and Linux:
pip install numpy scipy matplotlib
```

Set up the `cops_and_robots` code: 

```
cd ~ # or wherever you'd like to keep this code
git clone https://github.com/COHRINT/cops_and_robots.git
cd cops_and_robots
python setup.py install
```
Test out the code:
```
cd src/cops_and_robots
python main.py
```
