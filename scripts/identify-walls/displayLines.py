# Displays lines from an out.csv file generated without -DPRINT_INDICIES
# Note: This will print all lines in a file, if you want to limit it to 
# a specific frame, grep the .csv file for that frame name

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mcol

# Get filename
parser = argparse.ArgumentParser(
    prog="displayLines",
    description="Display lines from line CSV."
)
parser.add_argument("csvFile", help="Path of .csv frame file to display.")
parser.add_argument("--outFile", help="Path to write output figure to.")
parser.add_argument("--title", help="Title for plot")
args = parser.parse_args()

with open(args.csvFile, "r") as file:
    csvReader = csv.reader(file)
    data = np.array(list(csvReader))

    p1 = data[:,[1,2]].astype(np.float32)
    p2 = data[:,[3,4]].astype(np.float32)

    lines = list(map(lambda x, y: [x, y], list(map(tuple, p1)), list(map(tuple, p2))))
    colors = list(map(lambda line: mcol.hsv_to_rgb([np.random.rand(), 1, 0.75]), lines))

    print("Displaying {} lines.".format(len(lines)))

    lc = mc.LineCollection(lines, linewidths=2, colors=colors)    

    fig, ax = plt.subplots()
    ax.add_collection(lc)
    marker = ax.scatter(0, 0, marker="*", color="red")

    ax.autoscale()
    ax.margins(0.1)

    ax.set_title(args.title)

    ax.set_xlabel('x (m) - Relative to LiDAR')
    ax.set_ylabel('y (m) - Relative to LiDAR')

    

    ax.legend([marker], ["LiDAR"])
    
    # Save the file if outFile specified, otherwise display it
    if(args.outFile is not None):
        plt.savefig(args.outFile, dpi=300)
    else:
        plt.show()

    