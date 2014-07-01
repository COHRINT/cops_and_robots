from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    name='cops_and_robots',
    author='Nick Sweet',
    author_email='nick.sweet@colorado.edu',
    packages=['cops_and_robots'],
	package_dir={'':'src'},    
    include_package_data=True,
    platforms='any',
    requires=['std_msgs','rospy'],
    tests_require=['pytest'],
    install_requires=['getch>=1.0'],
    scripts=['scripts'],
)

setup(**setup_args)