SHELL := /bin/bash

# Example: make run binFile=../sample_input/zero_deg_only/ehall_hallway/bin/frame_0.zeroDeg.bin

run:
	mkdir -p out/
	(sbatch --wait run.sh && echo "Output:" && cat ZeroDegree.out)

run-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DCPU_IMPLEMENTATION -Wall -O3 -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binFile)
	
display-overlayed-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DCPU_IMPLEMENTATION -DPRINT_INDICES -Wall -O3 -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binFile)
	python3 ../scripts/zero-degree/displayOverlayed.py out/out.csv $(binFile) --outFile out/overlayed.png

display-overlayed-no-merge-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -DCPU_IMPLEMENTATION -DSKIP_SEGMENT_MERGING -Wall -O3 -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binFile)
	python3 ../scripts/zero-degree/displayOverlayed.py out/out.csv $(binFile) --outFile out/overlayed-no-merge.png

display-r-squared-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_R_SQUARED -DCPU_IMPLEMENTATION -DSKIP_SEGMENT_MERGING -Wall -O3 -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binFile)
	python3 ../scripts/zero-degree/displayRSquared.py out/out.csv --outFile out/r-squared.png
