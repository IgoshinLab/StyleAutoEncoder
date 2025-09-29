#!/bin/bash
#SBATCH --job-name=styleDAE-e13-post
#SBATCH --partition=oi1
##SBATCH --reservation=oi1
#SBATCH --account=oi1
#SBATCH --ntasks=1
#SBATCH --mail-user=jz79@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=/scratch-new/jz79/Welch/training-runs/run_e13post_debug
#SBATCH --mem=250G
#SBATCH --cpus-per-task=30
#SBATCH --threads-per-core=1
## srun --pty --partition=oi1 --account=oi1 --gres=gpu:1 --time=00:30:00 --mem=100G --cpus-per-task=20 $SHELL

ml purge
ml GCC/11.2.0
ml Anaconda3/2022.05
#source ~/.bashrc
source activate qj3-dae3
mkdir /scratch-new/jz79/Welch/training-runs/e13-post
export OMP_NUM_THREADS=12

python train_Myxo.py --resume /scratch-new/jz79/Welch/training-runs/e13-post/00000-stylegan2-trainingset2-gpus4-batch112-gamma10/network-snapshot-001512.pkl --data /scratch-new/jz79/Welch/trainingset2 --label-dict /scratch-new/jz79/Welch/training_dict.pkl --outdir /scratch-new/jz79/Welch/training-runs/e13-post --gpus 4 --mbstd-group 4 --batch 112 --z-dim 13 --workers 56 --tick 2 --snap 50 --glr 0.001 --dlr 0.001 --brightness-norm 0 --resize-by 1
## python train_Myxo.py --data /home/xavier/Documents/dataset/Welch/trainingset2/pre-training.zip --gpus 1 --mbstd-group 4 --batch 4 --z-dim 13 --workers 4 --tick 1 --snap 1 --glr 0.001 --dlr 0.001 --brightness-norm 0 --resize-by 1.15
