import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# From page 25 of data sheet
scan_layer_angles = [
    -22.2,
    -17.2,
    -12.3,
    -7.3,
    -2.5,
    0,      # High resolution
    2.2,
    7.0,
    12.9,
    17.2,
    21.8,
    26.6,
    31.5,
    34.2,  # High resolution
    36.7,
    42.2
]

def pointsFromData(data):
    return np.column_stack((data["points_x"], data["points_y"], data["points_z"]))

def getAzimuth(point):
    r = math.sqrt(point[0]**2 + point[1]**2)
    return math.degrees(math.atan2(point[2], r))

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def getScanLayer(azimuth):
    closestIndex = min(range(len(scan_layer_angles)), key = lambda i: abs(scan_layer_angles[i]-azimuth))
    difference = abs(scan_layer_angles[closestIndex] - azimuth)
    if(difference > 1):
        raise Exception("Difference is greater than 1!")
    
    # Scan layer number is index + 1 since scan layers start at 1
    return closestIndex + 1

def plotPoints(points, title):
    fig = plt.figure()
    fig.canvas.manager.set_window_title(title)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(min(np.min(points[:,2]), -1), max(np.max(points[:,2]), 10))
    plt.show()

def plotLayersSeperately(points):
    for layer in range(1, 17):
        fig = plt.figure()
        fig.canvas.manager.set_window_title("Layer {}".format(layer))

        ax = fig.add_subplot(projection='3d')
        
        for point in points:
            scanLayer = getScanLayer(getAzimuth(point))
            if(scanLayer == layer):
                ax.scatter(point[0], point[1], point[2], c=colors.hsv_to_rgb([scanLayer/16, 1, 1]))

        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(np.min(points[:,0]), np.max(points[:,0]))
        ax.set_ylim(np.min(points[:,1]), np.max(points[:,1]))
        ax.set_zlim(np.min(points[:,2]), np.max(points[:,2]))
    
    plt.show()    