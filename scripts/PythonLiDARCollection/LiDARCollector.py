import argparse
from sick_scan_api import *
from LiDARAPIHandler import *
import time

parser = argparse.ArgumentParser(
    prog="LiDARCollector",
    description="Collects 3D LiDAR Data from a SICK Sensor."
)

parser.add_argument("--launch-file", required=True, help="Path of launch file within the sick_scan_Xd repo")
parser.add_argument("--lidar-ip", required=True, help="IP address of the LiDAR")
parser.add_argument("--receiver-ip", required=True, help="IP of the receiver (must be on same subnet as LiDAR)")

def cartesianPointcloudCallback(apiHandle, msg):
    print("{}x{} pointcloud message received".format(msg.contents.width, msg.contents.height))

if __name__ == "__main__":
    args = parser.parse_args()
    scanLibrary, apiHandle = openLidarAPI(args.launch_file, args.lidar_ip, args.receiver_ip)

    callbackHandle = SickScanPointCloudMsgCallback(cartesianPointcloudCallback)
    SickScanApiRegisterCartesianPointCloudMsg(scanLibrary, apiHandle, callbackHandle)

    time.sleep(10)

    SickScanApiDeregisterCartesianPointCloudMsg(scanLibrary, apiHandle, callbackHandle)

    closeLidarAPI(scanLibrary, apiHandle)