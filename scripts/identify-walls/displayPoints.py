# Plots the points in a zero-degree .bin file

import argparse
from zeroDegreeBinUtils import *
import matplotlib.pyplot as plt

# Get filename
parser = argparse.ArgumentParser(
    prog="displayPoints",
    description="Display points from a zero-degree frame .bin file."
)

parser.add_argument("frameBinFile", help="Path of .bin frame file to display.")
parser.add_argument("--outFile", help="Path to write output figure to.")
parser.add_argument("--title", help="Title of chart.")

args = parser.parse_args()

points = readBinPoints(args.frameBinFile)
print("Read {} points.".format(len(points)))

fig, ax = plt.subplots()

ax.scatter(points[:,0], points[:,1], color=[0, 0, 0, 0.1])
ax.scatter(0, 0, marker="*", color="red", s=150)

ax.autoscale()
ax.margins(0.1)

ax.set_xlabel('x (m) - Relative to LiDAR')
ax.set_ylabel('y (m) - Relative to LiDAR')

ax.legend(["Points (n={})".format(len(points)), "LiDAR"])

ax.set_title(args.title)

# Save the file if outFile specified, otherwise display it
if(args.outFile is not None):
    plt.savefig(args.outFile, dpi=300)
else:
    plt.show()