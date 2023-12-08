import numpy as np
import csv
import struct

def readBinPoints(filePath):
    with open(filePath, "rb") as frameBinFile:
        binaryData = frameBinFile.read()
        numPoints = struct.unpack("<I", binaryData[0:4])[0]

        if(4 + numPoints * (4 * 3) != len(binaryData)):
            raise Exception("Mismatch in file size.")

        pointData = struct.unpack("<I{}f".format(numPoints * 3), binaryData)
        
        points_x = np.array(pointData[1:numPoints + 1])
        points_y = np.array(pointData[numPoints + 1:2*numPoints + 1])

        points = np.column_stack((points_x, points_y))
    return points

def getLinesFromIndexFile(indexFilePath, points):
    with open(indexFilePath, "r") as lineSegmentCsvFile:
        csvReader = csv.reader(lineSegmentCsvFile)
        data = np.array(list(csvReader))
        lineSegmentIndices = data[:,[0,1]].astype(np.uint)

        lines = list(map(lambda indices: [
                tuple(points[indices[0]]),
                tuple(points[indices[1]])],
                lineSegmentIndices))
    
    return lines, lineSegmentIndices