#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -t 48:00:00
#SBATCH --mem=4G

#SBATCH -J TNUF1_DeepMellow

module load tensorflow/1.5.0_gpu_py3
module load cuda/9.0.176
module load cudnn/7.0
module load opengl/mesa-12.0.6
module load ffmpeg/4.0.1

for i in {1..50}
do
    /users/sk99/myenv/bin/python3 -u dqn-LunarLander-v2-mellow.py $i 1
    /users/sk99/myenv/bin/python3 -u dqn-LunarLander-v2-mellow.py $i 2
    /users/sk99/myenv/bin/python3 -u dqn-LunarLander-v2-mellow.py $i 5
done
