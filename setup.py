from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
import io
import codecs
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.txt')

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)

setup(
    name='cops_and_robots',
    version=0.1,#cops_and_robots.__version__,
    url='http://github.com/COHRINT/cops_and_robots/',
    license='Apache Software License',
    author='Nick Sweet',
    author_email='nick.sweet@colorado.edu',
    description='Dynamic target-tracking using iRobot Creates',
    long_description=long_description,
    packages=find_packages(exclude="test"),
	package_dir={'':'cops_and_robots'},    
    include_package_data=True,
    platforms='any',
    tests_require=['pytest'],
    install_requires=['getch>=1.0'],
    scripts=['scripts'],
    # cmdclass={'test': PyTest},
    # test_suite='cops_and_robots.test.test_cops-and-robots',
    # extras_require={
    #     'testing': ['pytest'],
    # }
)