#!/bin/bash

# Gazebo Setting

# launch rosbridge
xterm -e roslaunch rosbridge_server rosbridge_websocket.launch &

# launch vicon bridge
xterm -e roslaunch web_video_server web_video_server.launch & 


# launch gazebo world
xterm -e roslaunch gazebo_ros plain_gazebo_clue_world.launch &

#Start gzweb
./gzweb/start_gzweb.sh 