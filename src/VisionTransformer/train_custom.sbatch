#!/bin/bash
#SBATCH --job-name=exercise02
#SBATCH --cpus-per-task=4
#SBATCH --partition=test
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=0-00:10:00
#SBATCH --output=tcml_log/train_log_%j.out       
#SBATCH --error=tcml_log/train_log_%j.err
#SBATCH --mail-user=niklas.munkes@student.uni-tuebingen.de
  

cp -R /home/stud138/RecAndGenANNs_WS24/exercisesheet02/VisionTransformer/gameplay-data.hdf5 /scratch/$SLURM_JOB_ID/
cp -R /home/stud138/RecAndGenANNs_WS24/exercisesheet02/VisionTransformer/config/vit.json /scratch/$SLURM_JOB_ID/

singularity exec --nv //home/stud138/RecAndGenANNs_WS24/tcml_containers/ex02_ViT.simg python -m model.main -cfg config/vit.json --train #/scratch/$SLURM_JOB_ID/
