#!/bin/bash
#SBATCH --job-name="vs" # job-name will be used in ~/.ssh/config above
#SBATCH --partition=a100-galvani
#SBATCH --time=9:00:00 # walltime
#SBATCH --gpus=3 # for example request one default gpu
#SBATCH --output=src/galvani-log/vs-node-login.out
#SBATCH --error=src/galvani-log/vs-node-login.err
/usr/sbin/sshd -D -p 7122 -f /dev/null -h ${HOME}/ssh_keys/node_key # uses the user key as the host key