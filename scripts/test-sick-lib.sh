#!/usr/bin/env bash

# Configure based on sensor settings
SENSOR_IP=192.168.1.1   # Configured IP of the sensor
SENSOR_INTERFACE=eno1   # Interface ethernet cable of sensor is plugged into

FINAL_PROJECT_PATH="$(git rev-parse --show-toplevel)/FinalProject"
BUILD_PATH="$FINAL_PROJECT_PATH/build/SICK"
SICK_SCAN_XD_PATH="$FINAL_PROJECT_PATH/lib/sick_scan_xd"
RECEIVING_IP="$(ifconfig $SENSOR_INTERFACE | grep "inet " | awk '{print $2}')"

# Add SICK build path to LD_LIBRARY_PATH
LD_LIBRARY_PATH=.:$BUILD_PATH:$LD_LIBRARY_PATH

# Test passes as long as there are no errors, press enter to exit
$BUILD_PATH/sick_scan_xd_api_test $SICK_SCAN_XD_PATH/launch/sick_multiscan.launch hostname:=$SENSOR_IP udp_receiver_ip:=$RECEIVING_IP
