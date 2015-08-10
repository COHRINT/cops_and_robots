#!/bin/bash

# Gazebo Setting

# roscore
xterm -e roscore &

sleep 2 

# launch rosbridge
xterm -e roslaunch rosbridge_server rosbridge_websocket.launch &

# launch vicon bridge
xterm -e roslaunch web_video_server web_video_server.launch & 

# launch gazebo world
xterm -e roslaunch gazebo_ros plain_gazebo_clue_world.launch &

#Start gzweb
bash /home/sierra/gzweb/start_gzweb.sh
