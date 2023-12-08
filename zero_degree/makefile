SHELL := /bin/bash

# Example: make run binFile=../sample_input/zero_deg_only/ehall_hallway/bin/frame_0.zeroDeg.bin

run:
	g++ planeExtraction.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -Wall -O3 -std=c++17 -o planeExtraction
	./planeExtraction $(binFile) > out.csv
	
display-overlayed:
	g++ planeExtraction.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -Wall -O3 -std=c++17 -o planeExtraction
	python3 ../scripts/zero-degree/displayOverlayed.py <(./planeExtraction $(binFile)) $(binFile)

display-overlayed-no-merge:
	g++ planeExtraction.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -DSKIP_SEGMENT_MERGING -Wall -O3 -std=c++17 -o planeExtraction
	python3 ../scripts/zero-degree/displayOverlayed.py <(./planeExtraction $(binFile)) $(binFile)

display-r-squared:
	g++ planeExtraction.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_R_SQUARED -DSKIP_SEGMENT_MERGING -Wall -O3 -std=c++17 -o planeExtraction
	python3 ../scripts/zero-degree/displayRSquared.py <(./planeExtraction $(binFile))