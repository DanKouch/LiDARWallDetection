import argparse
import os

parser = argparse.ArgumentParser(
    prog="generateFrameList",
    description="Generates a frame list for a given directory."
)
parser.add_argument("directoryPath", help="Path of directory with .bin files to create listing of.")
args = parser.parse_args()

if(not os.path.isdir(args.directoryPath)):
    print("Error: Provided path is not a directory")

dirScan = os.scandir(args.directoryPath)
binFiles = []

for entry in dirScan:
    if entry.is_file():
        if(entry.name.endswith(".bin")):
            binFiles.append(entry.name)

# Sort by frame number
binFiles.sort(key = lambda name: int(name.split("_")[1].split(".")[0]))

for file in binFiles:
    print(file)