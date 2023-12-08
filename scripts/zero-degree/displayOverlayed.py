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
    prog="displayOverlayed",
    description="Display line segment CSV and points together."
)
parser.add_argument("lineSegmentCsvFile", help="Path of .csv line segment file to display.")
parser.add_argument("frameBinFile", help="Path of .bin frame file to display.")

args = parser.parse_args()

points = readBinPoints(args.frameBinFile)
print("Read {} points.".format(len(points)))

lines = getLinesFromIndexFile(args.lineSegmentCsvFile, points)
print("Read {} lines.".format(len(lines)))

colors = list(map(lambda line: mcol.hsv_to_rgb([np.random.rand(), 1, 1]), lines))
lc = mc.LineCollection(lines, linewidths=2, colors=colors)    

fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1], color=[0, 0, 0, 0.1])
ax.add_collection(lc)
ax.scatter(0, 0, marker="*", color="red", s=150)

for i, line in enumerate(lines):
    ax.annotate("{}".format(i), line[0])

ax.autoscale()
ax.margins(0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.savefig("overlayed.png")