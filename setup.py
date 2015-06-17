from setuptools import setup, find_packages
# from catkin_pkg.python_setup import generate_distutils_setup
import ez_setup
ez_setup.use_setuptools()

setup(
    # Author information and Metadata
    name='cops_and_robots',
    version=1.0,
    license='Apache Software License',
    author='Nick Sweet',
    author_email='nick.sweet@colorado.edu',

    # Package data
    packages=find_packages('src'),
    package_dir={'cops_and_robots':'src/cops_and_robots'},
    include_package_data=True,
    platforms='any',
    requires=['std_msgs','rospy'],
    tests_require=['pytest'],
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
)
