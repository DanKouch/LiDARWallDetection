# Displays R-squared data for standard output of single-frame
# runs compiled with -DPRINT_R_SQUARED.

import argparse
import numpy as np
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt

# Get filename
parser = argparse.ArgumentParser(
    prog="displayRSquared",
    description="Display R-Squared CSV."
)
parser.add_argument("csvFile", help="Path of .csv frame file to display.")
parser.add_argument("--outFile", help="Path to write output figure to.")
parser.add_argument("--title", help="Title of chart")
args = parser.parse_args()

threshold = 0.55

with open(args.csvFile, "r") as file:
    csvReader = csv.reader(file)
    data = np.array(list(csvReader))
    x = data[:,0].astype(np.float32)
    y = data[:,1].astype(np.float32)
    z = data[:,2].astype(np.float32)
    rSquared = data[:,3].astype(np.float32)
    n = data[:,4]

    fig = plt.figure()
    fig.canvas.manager.set_window_title("CSV Data")
    ax = fig.add_subplot()

    colormap = mpl.colormaps["winter"]

    def getColor(rS):
        c = colormap(rS)
        a = 0.1 + 0.9 * (1-min(rS, 1))
        return (c[0], c[1], c[2], a)

    colors = list(map(getColor, rSquared))

    g = ax.scatter(x, y, c=colors)
    ax.scatter(0, 0, marker="*", color="red", s=150)

    ax.set_xlabel('x (m) - Relative to LiDAR')
    ax.set_ylabel('y (m) - Relative to LiDAR')

    ax.set_title(args.title)

    plt.colorbar(mpl.cm.ScalarMappable(cmap=colormap), ax=ax)

    # Save the file if outFile specified, otherwise display it
    if(args.outFile is not None):
        plt.savefig(args.outFile, dpi=300)
    else:
        plt.show()
