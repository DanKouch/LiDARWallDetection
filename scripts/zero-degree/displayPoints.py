import argparse
import numpy as np
import csv
import math
from zeroDegreeBinUtils import *
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mcol
import struct
import sys

# Get filename
parser = argparse.ArgumentParser(
    prog="displayPoints",
    description="Display points from a zero-degree frame .bin file."
)

parser.add_argument("frameBinFile", help="Path of .bin frame file to display.")

args = parser.parse_args()

points = readBinPoints(args.frameBinFile)
print("Read {} points.".format(len(points)))

fig, ax = plt.subplots()

ax.scatter(points[:,0], points[:,1], color=[0, 0, 0, 0.1])
ax.scatter(0, 0, marker="*", color="red", s=150)

ax.autoscale()
ax.margins(0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.savefig("points.png")