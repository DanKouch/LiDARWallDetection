import argparse
from sick_scan_api import *
from LiDARAPIHandler import *
import time

parser = argparse.ArgumentParser(
    prog="LiDARCollector",
    description="Collects 3D LiDAR Data from a SICK Sensor."
)

parser.add_argument("--launch-file", required=True, help="Path of launch file within the sick_scan_Xd repo")
parser.add_argument("--lidar-ip", required=True, help="IP address of the LiDAR")
parser.add_argument("--receiver-ip", required=True, help="IP of the receiver (must be on same subnet as LiDAR)")
parser.add_argument("--collection-time", required=True, help="How many seconds to collect data for")

frame = 0

# Note: Function largely taken from the following SICK-provided usage example:
# https://github.com/SICKAG/sick_scan_xd/blob/master/doc/sick_scan_api/sick_scan_api.md#usage-example
def cartesianPointcloudCallback(apiHandle, msg):
    global frame
    print("{}x{} pointcloud message received".format(msg.contents.width, msg.contents.height))

    pointcloud_msg = msg.contents

    # Extract timestamp metadata
    timestamp = np.array([msg.contents.header.timestamp_sec, msg.contents.header.timestamp_nsec])

    # Extract field metadata for x, y, and z
    num_fields = pointcloud_msg.fields.size
    msg_fields_buffer = pointcloud_msg.fields.buffer
    field_offset_x = -1
    field_offset_y = -1
    field_offset_z = -1
    for n in range(num_fields):
        field_name = ctypesCharArrayToString(msg_fields_buffer[n].name)
        print(field_name)
        field_offset = msg_fields_buffer[n].offset
        if field_name == "x":
            field_offset_x = msg_fields_buffer[n].offset
        elif field_name == "y":
            field_offset_y = msg_fields_buffer[n].offset
        elif field_name == "z":
            field_offset_z = msg_fields_buffer[n].offset

    # Extract point data into numpy arrays
    cloud_data_buffer_len = (pointcloud_msg.row_step * pointcloud_msg.height) # length of polar cloud data in byte
    assert(pointcloud_msg.data.size == cloud_data_buffer_len and field_offset_x >= 0 and field_offset_y >= 0 and field_offset_z >= 0)
    
    cloud_data_buffer = bytearray(cloud_data_buffer_len)
    for n in range(cloud_data_buffer_len):
        cloud_data_buffer[n] = pointcloud_msg.data.buffer[n]
    
    points_x = np.zeros(pointcloud_msg.width * pointcloud_msg.height, dtype = np.float32)
    points_y = np.zeros(pointcloud_msg.width * pointcloud_msg.height, dtype = np.float32)
    points_z = np.zeros(pointcloud_msg.width * pointcloud_msg.height, dtype = np.float32)
    
    point_idx = 0
    for row_idx in range(pointcloud_msg.height):
        for col_idx in range(pointcloud_msg.width):
            pointcloud_offset = row_idx * pointcloud_msg.row_step + col_idx * pointcloud_msg.point_step
            points_x[point_idx] = np.frombuffer(cloud_data_buffer, dtype = np.float32, count = 1, offset = pointcloud_offset + field_offset_x)[0]
            points_y[point_idx] = np.frombuffer(cloud_data_buffer, dtype = np.float32, count = 1, offset = pointcloud_offset + field_offset_y)[0]
            points_z[point_idx] = np.frombuffer(cloud_data_buffer, dtype = np.float32, count = 1, offset = pointcloud_offset + field_offset_z)[0]
            point_idx = point_idx + 1

    # Save numpy data into a npy file, and export each frame individually
    print("Saving frame {}...".format(frame))
    np.savez("frame_{}.npz".format(frame), points_x=points_x, points_y=points_y, points_z=points_z, timestamp=timestamp)
    frame = frame + 1

if __name__ == "__main__":
    args = parser.parse_args()
    scanLibrary, apiHandle = openLidarAPI(args.launch_file, args.lidar_ip, args.receiver_ip)

    callbackHandle = SickScanPointCloudMsgCallback(cartesianPointcloudCallback)
    SickScanApiRegisterCartesianPointCloudMsg(scanLibrary, apiHandle, callbackHandle)

    # Collect data for the requested amount of time
    time.sleep(float(args.collection_time))

    SickScanApiDeregisterCartesianPointCloudMsg(scanLibrary, apiHandle, callbackHandle)

    closeLidarAPI(scanLibrary, apiHandle)