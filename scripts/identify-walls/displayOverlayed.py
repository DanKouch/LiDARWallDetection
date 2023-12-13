# Displays lines from an out.csv file overlaid on the points of a
# data frame. Note that the out.csv file must be generated with
# -DPRINT_INDICIES.

import argparse
import numpy as np
from zeroDegreeBinUtils import *
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mcol

# Get filename
parser = argparse.ArgumentParser(
    prog="displayOverlayed",
    description="Display line segment CSV and points together."
)
parser.add_argument("lineSegmentCsvFile", help="Path of .csv line segment file to display.")
parser.add_argument("frameBinFile", help="Path of .bin frame file to display.")
parser.add_argument("--outFile", help="Path to write output figure to.")

args = parser.parse_args()

points = readBinPoints(args.frameBinFile)
print("Read {} points.".format(len(points)))

lines, lineSegmentIndices = getLinesFromIndexFile(args.lineSegmentCsvFile, points)
print("Read {} lines.".format(len(lines)))

colors = list(map(lambda line: mcol.hsv_to_rgb([np.random.rand(), 1, 1]), range(len(lines) + 1)))
colors[len(lines)] = [0, 0, 0]

def getLineFromIndex(idx):
    for i, r in enumerate(lineSegmentIndices):
        if(idx >= r[0] and idx <= r[1]):
            return i
    return len(lineSegmentIndices)

#colors = list(map(lambda line: mcol.hsv_to_rgb([np.random.rand(), 1, 1]), lines))
lc = mc.LineCollection(lines, linewidths=2, colors=colors)

def getPointColor(index):
    lineNumber = getLineFromIndex(index)
    return np.array(colors[lineNumber])/1.5

pointsColors = list(map(getPointColor, range(len(points))))

fig, ax = plt.subplots()
ax.scatter(points[:,0], points[:,1], color=pointsColors)
ax.add_collection(lc)
ax.scatter(0, 0, marker="*", color="red", s=150)

for i, line in enumerate(lines):
    ax.annotate("{}".format(i), line[0])

ax.autoscale()
ax.margins(0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')

# Save the file if outFile specified, otherwise display it
if(args.outFile is not None):
    plt.savefig(args.outFile)
else:
    plt.show()
