# Based on https://github.com/SICKAG/sick_scan_xd/blob/master/examples/python/minimum_sick_scan_api_client.py

from sick_scan_api import *

def openLidarAPI(launch_file, lidar_ip, receiver_ip):
    sick_scan_library = SickScanApiLoadLibrary(["build/", "build_linux/", "../../build/", "../../build_linux/", "./", "../"], "libsick_scan_xd_shared_lib.so")
    api_handle = SickScanApiCreate(sick_scan_library)
    SickScanApiInitByLaunchfile(sick_scan_library, api_handle, "{} hostname:={} udp_receiver_ip:={} custom_pointclouds:=cloud_unstructured_fullframe".format(launch_file, lidar_ip, receiver_ip))

    return sick_scan_library, api_handle

def closeLidarAPI(sick_scan_library, api_handle):
    SickScanApiClose(sick_scan_library, api_handle)
    SickScanApiRelease(sick_scan_library, api_handle)
    SickScanApiUnloadLibrary(sick_scan_library)
