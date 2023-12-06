# Extracts scan planes from an NPZ file

# Example usage python3 npzToBin.py *.npz

import argparse
import numpy as np
import os
import struct
import sys

parser = argparse.ArgumentParser(
    prog="npzToBin",
    description="Converts npz files to custom binary format."
)

parser.add_argument("frameFiles", nargs='+', help="Path of .npz frame file to process.")

args = parser.parse_args()

for path in args.frameFiles:
    if(not os.path.isfile(path)):
        raise Exception("'{}' does not exist!".format(path))

for frameFile in args.frameFiles:
    fileNameBase = os.path.basename(frameFile).split(".")
    fileExtension = fileNameBase.pop()
    fileNameBase = ".".join(fileNameBase)
    outputFileName = ".".join([fileNameBase, "bin"])

    # Read NPZ file
    data = np.load(frameFile, mmap_mode="r")

    # Get packed structure
    numPoints = len(data["points_x"])
    packed = struct.pack("<I{}f".format(numPoints*3), numPoints, *data["points_x"], *data["points_y"], *data["points_z"])

    # Write binary file
    with open(outputFileName, "wb") as file:
        file.write(packed)
    
    data.close()