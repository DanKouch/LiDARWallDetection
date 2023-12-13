SHELL := /bin/bash
numFrames := 1
binDir := ../sample_input/ehall_1800_back/bin

run:
	mkdir -p out/
	(sbatch --export=BIN_DIR='$(binDir)' --wait sbatch/run-gpu.sh && echo "Output:" && cat IdentifyWalls.out)

run-indicies:
	mkdir -p out/
	(sbatch --export=BIN_DIR='$(binDir)' --wait sbatch/run-gpu-indicies.sh && echo "Output:" && cat IdentifyWalls.out)

run-uvm:
	mkdir -p out/
	(sbatch --export=BIN_DIR='$(binDir)' --wait sbatch/run-gpu-uvm.sh && echo "Output:" && cat IdentifyWalls.out)

run-no-fast-math:
	mkdir -p out/
	(sbatch --export=BIN_DIR='$(binDir)' --wait sbatch/run-gpu-no-fast-math.sh && echo "Output:" && cat IdentifyWalls.out)

run-cpu:
	mkdir -p out/
	(sbatch --export=BIN_DIR='$(binDir)' --wait sbatch/run-cpu.sh && echo "Output:" && cat IdentifyWalls.out)
	
run-indicies-cpu:
	mkdir -p out/
	(sbatch --export=BIN_DIR='$(binDir)' --wait sbatch/run-cpu-indicies.sh && echo "Output:" && cat IdentifyWalls.out)
	

# Warning! Don't use the following on Euler, as they aren't wrapped in a sbatch script

display-overlayed-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ identifyWalls.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -Wall -O3 -ffast-math -std=c++17 -o identifyWalls
	./identifyWalls out/out.csv $(binDir) 1
	python3 ../scripts/identify-walls/displayOverlayed.py out/out.csv $(binDir)/frame_0.zeroDeg.bin --outFile out/overlayed$(comment).png

display-overlayed-no-merge-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ identifyWalls.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -DSKIP_SEGMENT_MERGING -Wall -O3 -ffast-math -std=c++17 -o identifyWalls
	./identifyWalls out/out.csv $(binDir) 1
	python3 ../scripts/identify-walls/displayOverlayed.py out/out.csv $(binDir)/frame_0.zeroDeg.bin --outFile out/overlayed-no-merge$(comment).png

display-r-squared-cpu:
	mkdir -p out/
	rm -f out/out.csv
	g++ identifyWalls.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_R_SQUARED -DSKIP_SEGMENT_MERGING -Wall -O3 -ffast-math -std=c++17 -o identifyWalls
	python3 ../scripts/identify-walls/displayRSquared.py <(./identifyWalls out/out.csv $(binDir) 1 | grep ", 0.0") --outFile out/r-squared$(comment).png
