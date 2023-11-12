# Based on https://github.com/SICKAG/sick_scan_xd/blob/master/examples/python/minimum_sick_scan_api_client.py


import os
import sys
import time
    
# Make sure sick_scan_api is searched in all folders configured in environment variable PYTHONPATH
def appendPythonPath():
    pythonpath = os.environ['PYTHONPATH']
    for folder in pythonpath.split(";"):
        sys.path.append(os.path.abspath(folder))

try:
    # import sick_scan_api
    from sick_scan_api import *
except ModuleNotFoundError:
    print("import sick_scan_api failed, module sick_scan_api not found, trying with importlib...")
    appendPythonPath()
    import importlib
    sick_scan_api = importlib.import_module("sick_scan_api")

def pyCustomizedPointCloudMsgCb(api_handle, msg):
    print("LiDAR point cloud message received: {} x {} pointcloud message received".format(msg.contents.width, msg.contents.height))

# Pass launchfile and commandline arguments to sick_scan_library
cli_args = " ".join(sys.argv[1:])

sick_scan_library = SickScanApiLoadLibrary(["build/", "build_linux/", "../../build/", "../../build_linux/", "./", "../"], "libsick_scan_xd_shared_lib.so")

api_handle = SickScanApiCreate(sick_scan_library)
SickScanApiInitByLaunchfile(sick_scan_library, api_handle, cli_args)

# Register for pointcloud messages
cartesian_pointcloud_callback = SickScanPointCloudMsgCallback(pyCustomizedPointCloudMsgCb)
SickScanApiRegisterCartesianPointCloudMsg(sick_scan_library, api_handle, cartesian_pointcloud_callback)

# Run application or main loop
time.sleep(10)

# Close lidar and release sick_scan api
SickScanApiDeregisterCartesianPointCloudMsg(sick_scan_library, api_handle, cartesian_pointcloud_callback)
SickScanApiClose(sick_scan_library, api_handle)
SickScanApiRelease(sick_scan_library, api_handle)
SickScanApiUnloadLibrary(sick_scan_library)
