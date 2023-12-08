#!/usr/bin/env zsh
#SBATCH --job-name=ZeroDegree
#SBATCH --partition=instruction
#SBATCH --time=00-00:1:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=ZeroDegree.out

cd $SLURM_SUBMIT_DIR

module load nvidia/cuda/12.2.0                                                                                                                                                                      
module load gcc/11.3.0

BIN_FILE=../sample_input/zero_deg_only/ehall_1800_back/bin/frame_0.zeroDeg.bin

mkdir -p out/

rm -f ./zeroDegree

nvcc zeroDegree.cpp fileHandler.cpp dataFrame.cpp gpuImplementation.cu -Xcompiler -Wall -Xcompiler -O3 -DPRINT_INDICES -lcublas -allow-unsupported-compiler -std=c++17 -o zeroDegree

./zeroDegree $BIN_FILE

