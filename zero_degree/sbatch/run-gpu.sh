#!/usr/bin/env zsh
#SBATCH --job-name=ZeroDegree
#SBATCH --partition=instruction
#SBATCH --time=00-00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=ZeroDegree.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/11.8.0
module load gcc/11.3.0

BIN_DIR=../sample_input/ehall_1800_back/bin/

mkdir -p out/
rm -f out/out.csv
rm -f ./zeroDegree

nvcc zeroDegree.cpp fileHandler.cpp dataFrame.cpp gpuImplementation.cu -Xcompiler -Wall -Xptxas -O3 -Xcompiler -O3 --use_fast_math -std=c++17 -o zeroDegree

# Ensure all input files are loaded into memory on our node
# This is the more realistic scenerio, as a LiDAR would be able to
# directly stream it's data to our application; we wouldn't need to
# wait for disk access.
for i in {0..3}
do
    ./zeroDegree /dev/null $BIN_DIR 0
done

./zeroDegree out/out.csv $BIN_DIR 0
