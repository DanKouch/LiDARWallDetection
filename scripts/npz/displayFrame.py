import argparse
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os

# Get filename
parser = argparse.ArgumentParser(
    prog="displayFrame",
    description="Display a lidar data frame."
)
parser.add_argument("frameFile", help="Path of .npz frame file to display.")
args = parser.parse_args()

data = np.load(args.frameFile, mmap_mode="r")

# Extract timestamp and points from file
timestamp = data["timestamp"][0]

points = np.column_stack((data["points_x"], data["points_y"], data["points_z"]))

# Add points to point cloud visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Add coordinate axes to origin (location of LiDAR)
coordinate_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

# Visualize in window
window_name = "{} - {}".format(os.path.basename(args.frameFile), timestamp)
o3d.visualization.draw_geometries([pcd, coordinate_axes], window_name=window_name)

# Close input data file
data.close()