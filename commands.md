## create new conda env from yml file
conda env create -f environment.yml

## update conda env from fileco
conda env update --file environment.yml --prune

## activate conda env 
conda activate citations

## remove conda env
conda remove -n citations --all

## interactive session
### Galvani a100
srun --job-name "InteractiveJob" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash

## get architecture/driver info 
nvidia-smi

##

