#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J sc_train                # Job name
#SBATCH --ntasks=2                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=2                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=1-00:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:8               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=../../galvani-log/log-a100-start_train2.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=../../galvani-log/log-a100-start_train2.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=niklas.munkes@student.uni-tuebingen.de   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
squeue -u "gwb204" -o"%.10i %.15P %.10u %.2t %.10M %.6D %C %n"

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls

# Compute Phase
srun sh start_train2.sh