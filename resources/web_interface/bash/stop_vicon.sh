#!/bin/bash

# Shut down robots

	# Security camera's
	xterm -e rosnode kill /cam1/usb_cam &

	xterm -e rosnode kill /cam2/usb_cam &

	xterm -e rosnode kill /cam3/usb_cam &
	
	# Deckard
	xterm -e rosnode kill /deckard/map_server &

	# Pris
	xterm -e rosnode kill /pris/map_server &
	
	# Roy
	xterm -e rosnode kill /roy/map_server &

	# Zhora
	xterm -e rosnode kill /zhora/map_server &

	# xterm
	killall xterm &

	# terminal
	killall gnome-terminal
