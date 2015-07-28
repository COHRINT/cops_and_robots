#!/bin/bash

# Gazebo Setting
echo "success" 

# launch rosbridge
gnome-terminal -e roslaunch rosbridge_server rosbridge_websocket.launch &

# launch vicon bridge
gnome-terminal -e roslaunch web_video_server web_video_server.launch & 

# launch gazebo world
gnome-terminal -e roslaunch gazebo_ros plain_gazebo_clue_world.launch 


#Start gzweb
# ./gzweb/start_gzweb.sh 