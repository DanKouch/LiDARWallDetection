SHELL := /bin/bash
numFrames := 1
binDir := ../sample_input/ehall_1800_back/bin

run:
	mkdir -p out/
	(sbatch --wait run-gpu.sh && echo "Output:" && cat ZeroDegree.out)

run-cpu:
	mkdir -p out/
	(sbatch --wait run-cpu.sh && echo "Output:" && cat ZeroDegree.out)
	
# Warning! Don't use the following on Euler, as they aren't wrapped in a sbatch script

display-overlayed-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -Wall -O3 -ffast-math -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binDir) 1
	python3 ../scripts/zero-degree/displayOverlayed.py out/out.csv $(binDir)/frame_0.zeroDeg.bin --outFile out/overlayed.png

display-overlayed-no-merge-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -DSKIP_SEGMENT_MERGING -Wall -O3 -ffast-math -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binDir) 1
	python3 ../scripts/zero-degree/displayOverlayed.py out/out.csv $(binDir)/frame_0.zeroDeg.bin --outFile out/overlayed-no-merge.png

display-r-squared-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_R_SQUARED -DSKIP_SEGMENT_MERGING -Wall -O3 -ffast-math -std=c++17 -o zeroDegree
	./zeroDegree out/out.csv $(binDir) 1
	python3 ../scripts/zero-degree/displayRSquared.py out/out.csv --outFile out/r-squared.png
