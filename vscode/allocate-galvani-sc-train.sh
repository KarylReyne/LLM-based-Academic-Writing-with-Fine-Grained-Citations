#!/bin/bash
#SBATCH --job-name="sc-train"
#SBATCH --partition=a100-galvani
#SBATCH --time=10:00:00
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=src/galvani-log/sc-train-node-login.out
#SBATCH --error=src/galvani-log/sc-train-node-login.err

conda activate /home/geiger/gwb204/miniconda3/envs/citations

/usr/sbin/sshd -D -p 7122 -f /dev/null -h ${HOME}/ssh_keys/node_key # uses the user key as the host key