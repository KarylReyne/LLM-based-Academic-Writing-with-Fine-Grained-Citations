#!/bin/bash
#SBATCH --job-name=exercise02
#SBATCH --cpus-per-task=4
#SBATCH --partition=test
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=0-00:05:00
#SBATCH --output=train_log_%j.out       
#SBATCH --error=train_log_%j.err        

rsync -av /common/share/LectureRecAndGenANNs/exercisesheet02/data/gameplay-data.hdf5 /scratch/$SLURM_JOB_ID/
rsync -av /common/share/LectureRecAndGenANNs/container.sif /scratch/$SLURM_JOB_ID/

singularity run --bind $PWD:/mnt /scratch/$SLURM_JOB_ID/container.sif /mnt/train.sh /scratch/$SLURM_JOB_ID/
