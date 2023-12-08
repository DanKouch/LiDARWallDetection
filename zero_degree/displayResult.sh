#!/usr/bin/env bash
g++ planeExtraction.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -Wall -O3 -std=c++17 -o planeExtraction
python3 ../scripts/displayCombined.py <(./planeExtraction $1) $1