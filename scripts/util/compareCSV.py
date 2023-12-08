# Checks for discrepancies between two CSV files

import argparse
import numpy as np
import csv
import math
import matplotlib.pyplot as plt
import sys

# Get filename
parser = argparse.ArgumentParser(
    prog="compareCSV",
    description="Compare CSV."
)
parser.add_argument("csvFile1", help="Path of first .csv frame file to compare.")
parser.add_argument("csvFile2", help="Path of second .csv frame file to compare.")
args = parser.parse_args()

with open(args.csvFile1, "r") as file1:
    csvReader1 = csv.reader(file1)
    data1 = np.array(list(csvReader1)).astype(np.float32)
    
    with open(args.csvFile2, "r") as file2:
        csvReader2 = csv.reader(file2)
        data2 = np.array(list(csvReader2)).astype(np.float32)

        if(data1.shape != data2.shape):
            print("CSV files have different shapes.")
            sys.exit(0)

        tolerance = 0.02

        for r in range(data1.shape[0]):
            for c in range(data1.shape[1]):
                diff = abs(data1[r, c] - data2[r, c])
                if(diff > tolerance):
                    print("Mismatch at row {}, column {} by {}.".format(r, c, diff))