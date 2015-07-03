#!/usr/bin/env python

from cops_and_robots.robo_tools.pose import Pose

deckard = Pose([0, 0, 0], pose_source='odom')

deckard.pose = [1, 1, 1]

print(deckard.pose)

deckard.pose[1] = 2

print(deckard.pose)