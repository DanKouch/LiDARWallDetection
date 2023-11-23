#!/usr/bin/env bash

FINAL_PROJECT_PATH="$(git rev-parse --show-toplevel)/FinalProject"
BUILD_PATH="$FINAL_PROJECT_PATH/build/SICK"
SICK_SCAN_XD_PATH="$FINAL_PROJECT_PATH/lib/sick_scan_xd"

export LD_LIBRARY_PATH=$BUILD_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=$SICK_SCAN_XD_PATH/python/api:$PYTHONPATH

# Configuration
SENSOR_INTERFACE=eno1   # Interface ethernet cable of sensor is plugged into
LIDAR_LAUNCH_FILE=$SICK_SCAN_XD_PATH/launch/sick_multiscan.launch
LIDAR_RECEIVING_IP="$(ifconfig $SENSOR_INTERFACE | grep "inet " | awk '{print $2}')"

python3 LiDARCollector.py --launch-file=$LIDAR_LAUNCH_FILE --lidar-ip=192.168.1.1 --receiver-ip=$LIDAR_RECEIVING_IP