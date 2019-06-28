#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -t 48:00:00
#SBATCH --mem=16G

module load tensorflow/1.5.0_gpu_py3
module load cuda/9.0.176
module load cudnn/7.0
module load opengl/mesa-12.0.6
module load ffmpeg/4.0.1

/users/sk99/myenv/bin/python3 -u main.py --network_header_type=nature --env_name=Seaquest-v0 --t_target_q_update_freq=0.0001 --use_gpu=True --mellowmax=True --w=30

