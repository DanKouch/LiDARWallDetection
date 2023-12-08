import argparse
import numpy as np
from lidarDataUtil import *

# Get filename
parser = argparse.ArgumentParser(
    prog="displayFrame",
    description="Display a lidar data frame."
)
parser.add_argument("frameFile", help="Path of .npz frame file to display.")
args = parser.parse_args()

data = np.load(args.frameFile, mmap_mode="r")

points = np.column_stack((data["points_x"], data["points_y"], data["points_z"]))
plotPoints(points, "[Timestamp: {}] ({} points)".format(data["timestamp"][0], points.shape[0]))

data.close()