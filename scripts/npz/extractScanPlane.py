# Extracts scan planes from an NPZ file

# Example usage python3 extractScanPlane.py 6 zeroDeg *.npz

# Note:
# Since this script has this equivilant functionality of disabling all other
# scan planes in LiDAR configuration, preprocessing done using this script
# shouldn't be accounted for in timing.

import argparse
import numpy as np
import os
from lidarDataUtil import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

parser = argparse.ArgumentParser(
    prog="extractScanPlanes",
    description="Extracts lidar scan planes."
)

parser.add_argument("scanPlanes", help="Comma separated plane numbers (starting at 1)")
parser.add_argument("fileNameComment", help="String in file name.")
parser.add_argument("frameFiles", nargs='+', help="Path of .npz frame file to process.")

args = parser.parse_args()

# Validate argument
scanPlanes = list(map(lambda num: int(num), args.scanPlanes.split(",")))
for plane in scanPlanes:
    if(plane < 0 or plane > len(scan_layer_angles) - 1):
        raise Exception("'{}' is not a valid scan plane number!".format(plane))

for path in args.frameFiles:
    if(not os.path.isfile(path)):
        raise Exception("'{}' does not exist!".format(path))

for frameFile in args.frameFiles:
    fileNameBase = os.path.basename(frameFile).split(".")
    fileExtension = fileNameBase.pop()
    fileNameBase = ".".join(fileNameBase)

    outputFileName = ".".join([fileNameBase, args.fileNameComment, fileExtension])

    data = np.load(frameFile, mmap_mode="r")
    points = pointsFromData(data)

    points_x = np.array([])
    points_y = np.array([])
    points_z = np.array([])

    for point in points:
        scanLayer = getScanLayer(getAzimuth(point))
        if(scanLayer in scanPlanes):
            points_x = np.append(points_x, point[0])
            points_y = np.append(points_y, point[1])
            points_z = np.append(points_z, point[2])
    
    extractedPoints = np.column_stack((points_x, points_y, points_z))
    
    np.savez(outputFileName, points_x=points_x, points_y=points_y, points_z=points_z, timestamp=data["timestamp"])
    
    data.close()