#!/usr/bin/env zsh
#SBATCH --job-name=ZeroDegree
#SBATCH --partition=instruction
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=ZeroDegree.out

cd $SLURM_SUBMIT_DIR

module load gcc/11.3.0

BIN_DIR=../sample_input/ehall_1800_back/bin/

mkdir -p out/
rm -f out/out.csv
rm -f ./zeroDegree

g++ zeroDegree.cpp fileHandler.cpp dataFrame.cpp cpuImplementation.cpp -DPRINT_INDICES -Wall -O3 -ffast-math -std=c++17 -o zeroDegree

# Ensure all input files are loaded into memory on our node
# This is the more realistic scenerio, as a LiDAR would be able to
# directly stream it's data to our application; we wouldn't need to
# wait for disk access.
for i in {0..3}
do
    ./zeroDegree /dev/null $BIN_DIR 0 > /dev/null
done

./zeroDegree out/out.csv $BIN_DIR 0