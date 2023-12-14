# GPU-Accelerated Real-Time LiDAR Plane Detection

## About
This program takes in multiple frames of zero-degree scan angle LiDAR data from a SICK multiScan 100 scanner, and outputs the walls (2D line segments in x and y coordinates) it detects for each frame. We've implemented our algorithm in both CPU and GPU code for the sake of comparision, which each have their respective `make` rules. The program's code is in `identify_walls/`, everything else is supporting files for collecting LiDAR data from the scanner and displaying that data, as well as the program's output.

## Running The Program
1. Ensure you have git lfs installed (It's installed on Euler)
2. Download repository to Euler
3. If using our collected LiDAR data (which you should be), decompress `sample_input.tgz`
4. Change directories to `identify-walls/`
5. Run `make run binDir=BIN_DIR` to run our GPU implementation
6. Run `make run-cpu binDir=BIN_DIR` to run our CPU implementation

`BIN_DIR` can be any of the following for our sample input:
- `../sample_input/ehall_1800_front/bin/` for a scan from the front of 1800 Engineering Hall
- `../sample_input/ehall_1800_back/bin/` for a scan from the back of 1800 Engineering Hall
- `../sample_input/ehall_hallway/bin/` for a scan from a hallway on the second floor of Engineering Hall

After the above steps have been completed, timing information will be displayed in IdentifyWalls.out, and the program's output will be placed in `out/out.csv`, where output.csv contains a line for every wall identified of the following format:
```csv
frameFileName, lineStartingX, lineStartingY, lineEndingX, lineEndingY 
```

### Displaying Output Data

`out/out.csv` contains line information for each frame. To display a single frame of data in Matplotlib (and save it to `out/lines.png`), run the following command (replace `frame_0` with `frame_#`, where # is your desired frame): 

```bash
python3 ../scripts/identify-walls/displayLines.py --outFile=out/lines.png <(grep "frame_0" out/out.csv)
```

### Displaying Overlayed Output Data

By running the `run-indicies` and `run-indicies-cpu` make rules instead, a CSV will be generated of the following format:
```csv
frameFileName, startingPointIdx, endingPointIdx
```

This allows the generated lines to be displayed on top of a scatter plot of the collected points with the `../scripts/identify-walls/displayOverlayed.py` script.

An example of this scripts usage is as follows:

```bash
python3 ../scripts/identify-walls/displayOverlayed.py --outFile=out/overlayed.png <(grep "frame_0" out/out.csv) ../sample_input/ehall_1800_back/bin/frame_0.zeroDeg.bin
```

## Folder Layout
- `identify_walls/` contains our program code, as well as sbatch scripts which can be used to run our program on Euler
- Once `sample_input.tgz` is opened, `sample_input/` contains collected LiDAR data which can be used as input for the program
- `lib/` contains submodules which are necessary for building the SICK scanner API. These submodules are downloaded and built by `scripts/build-sick-lib.sh`, *which is only necessary if you are collecing LiDAR data yourself*
- `documents/` contains documentation pertaining to the SICK multiScan 100 LiDAR
- `scripts/`
    - `scripts/identify-walls/` contains scripts used to display the output of the identify walls program
    - `scripts/npz` contains scripts used to display and process pointcloud `.npz` frames
    - `scripts/PythonLiDARCollection` contains scripts used to collect data frames from the LiDAR
    - `scripts/util` contain miscellaneous utility scripts

## LiDAR Data Collection

LiDAR pointcloud data is collected from a SICK multiScan 100 LiDAR.

### Sample Input

Since you likely do not have a SICK multiScan 100 LiDAR, we've collected sample input from 3 locations, which we've placed in `sample_input.tgz`. To use this data when running the identify walls program, untar and decompress this file.

Once this is done, each location has its own folder witin `sample_input/`, with subfolders for the zero-degree `.npz` files, and the `.zeroDeg.bin` files which are used as input in the identify walls program. Note that each `bin/` subdirectory contains a file named frameList.txt, which contains each frame file in order (and is used as input to our program).

### LiDAR Data Collection Setup Steps
1. Run `scripts/build-sick-lib.sh` to pull necessary submodules and build SICK libraries
2. Run `scripts/ufw-allow-lidar.sh` to permit LIDAR data to be received from the scanner
3. Plug your SICK multiScan scanner into your computer
4. Connect power to your SICK multiScan scanner
5. Once the lights on your LiDAR have turned green, navigate to `192.168.1.1` to access the web interface
3. Configure your SICK scanner with the default password for the operater account, and enable at least the zero-degree scan layer
4. Take note of the network interface your scanner is connected to, then set your own IP on that interface to something in the `192.168.1.*` subnet
5. Modify the top of `./scripts/PythonLiDARCollection/lidar.sh` to set how long you want data to be collected for and on what network interface the LiDAR is on

### LiDAR Data Collection
After following the LiDAR data collection setup steps above, navigate to `./scripts/PythonLiDARCollection/` and run `lidar.sh`. Frame `.npz` files will be populated in your current directory.

### LiDAR Frame Processing
Data frames are exported as Numpy `.npz` files for easy processing. These files can be operated on by scripts in the `scripts/npz` directory to display each frame, extract individual scan planes, and convert the `.npz` files to the `.bin` file format used by our program.

#### `.zeroDeg.bin` Format
Our program takes in data in our own `.zeroDeg.bin` file format.

This format includes only points extracted from the zero degree scan plane in our own `.bin` format. The first four bytes of our binary format are the number of points as a little-endian integer. The next (4\*numPoints) bytes are an array of the x coordinates of each point as little-endian floats. The next (4\*numPoints) after these are the y coordinates, and the next (4\*numPoints) after these are the z coordinates.

**Note: This file format is reminicent of the structure-of-arrays format with which the LiDAR outputs data. If we were to directly collect LiDAR data in our program, we would be processing essentially the same format. As such, we don't take the time it takes to convert the easy-to-process `.npz` format to our `.zeroDeg.bin` format**


#### LiDAR Frame `.npz` to `.zeroDeg.bin` Conversion

`.npz` files can be converted to `.zeroDeg.bin` files using `./scripts/npz/extractScanPlane.py` with the scan layer selected as 6 (the zero-degree layer), and then `./scripts/npz/npzToBin.py`. Extracting the zero-degree scan layer is equivilent to only enabling the 0 degree scan layer in the LiDARs settings, and thus can be ommited if this configuration was set.

The following is an example of this usage:
```bash
./scripts/npz/extractScanPlane.py 6 zeroDeg example.npz
./scripts/npz/npzToBin.py example.zeroDeg.npz
```