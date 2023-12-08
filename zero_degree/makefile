SHELL := /bin/bash

# Example: make run binFile=../sample_input/zero_deg_only/ehall_hallway/bin/frame_0.zeroDeg.bin

run:
	mkdir -p out/
	(sbatch --wait run.sh && echo "Output:" && cat out/ZeroDegree.out)

run-cpu:
	mkdir -p out/
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DCPU_IMPLEMENTATION -Wall -O3 -std=c++17 -o zeroDegree
	./zeroDegree $(binFile) > out/out.csv
	
display-overlayed-cpu:
	mkdir -p out/
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DCPU_IMPLEMENTATION -DPRINT_INDICES -Wall -O3 -std=c++17 -o zeroDegree
	python3 ../scripts/zero-degree/displayOverlayed.py <(./zeroDegree $(binFile)) $(binFile) --outFile out/overlayed.png

display-overlayed-no-merge-cpu:
	mkdir -p out/
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -DCPU_IMPLEMENTATION -DSKIP_SEGMENT_MERGING -Wall -O3 -std=c++17 -o zeroDegree
	python3 ../scripts/zero-degree/displayOverlayed.py <(./zeroDegree $(binFile)) $(binFile) --outFile out/overlayed-no-merge.png

display-r-squared-cpu:
	mkdir -p out/
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_R_SQUARED -DCPU_IMPLEMENTATION -DSKIP_SEGMENT_MERGING -Wall -O3 -std=c++17 -o zeroDegree
	python3 ../scripts/zero-degree/displayRSquared.py <(./zeroDegree $(binFile)) --outFile out/r-squared.png
