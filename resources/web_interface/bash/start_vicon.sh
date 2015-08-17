#!/bin/bash

# Test setting

# launch rosbridge
xterm -e roslaunch rosbridge_server rosbridge_websocket.launch &

# launch vicon bridge
xterm -e roslaunch vicon_bridge vicon.launch &
	
# launch gazebo world
roslaunch gazebo_ros plain_vicon_clue_world.launch &

# Set up device Camera's

	# Security camera's
	xterm -e ssh pi@cam1 'roslaunch usb_cam usb_cam-test.launch' &

	xterm -e ssh pi@cam2 'roslaunch usb_cam usb_cam-test.launch' &

	xterm -e ssh pi@cam3 'roslaunch usb_cam usb_cam-test.launch' &


	# Tyrell commandes 2
	xterm -e roslaunch ~/cops_and_robots/launch/vicon_sys.launch &
	xterm -e rosservice call /vicon_bridge/calibrate_segment deckard deckard 0 100 &
	

	# Deckard commands 2
	xterm -e ssh odroid@deckard 'roslaunch ~/cops_and_robots/launch/vicon_nav.launch' &

	# Tyrell commandes 3
	xterm -e roslaunch turtlebot_rviz_launchers view_navigation.launch &
	xterm -e roslaunch turtlebot_teleop keyboard_teleop.launch &

# To set up web interface
	# launch web video server
	xterm -e roslaunch web_video_server web_video_server.launch &


# Start gzweb
bash /home/cohirnt/gzweb/start_gzweb.sh 