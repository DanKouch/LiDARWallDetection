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

nvcc zeroDegree.cpp fileHandler.cpp dataFrame.cpp gpuImplementation.cu -Xcompiler -Wall -Xptxas -O3 -Xcompiler -O3 -DPRINT_INDICES --use_fast_math -std=c++17 -o zeroDegree

./zeroDegree out/out.csv $BIN_DIR 0
