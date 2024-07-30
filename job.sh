#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=00:45:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --mem=100G   # maximum memory per node
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="trainvoidspotter"
#SBATCH --mail-user=jackplum@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

python3 /home/jackplum/Documents/projects/yolov5/train.py --img 640 --epochs 1200 --data config.yaml --weights yolov5s.pt
