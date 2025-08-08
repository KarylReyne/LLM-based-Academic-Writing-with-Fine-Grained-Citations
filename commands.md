## create new conda env from yml file
conda env create -f environment.yml

## update conda env from fileco
conda env update --file environment.yml --prune

## activate conda env 
conda activate citations

## remove conda env
conda remove -n citations --all

## delete files starting with foo
### check what would get deleted
find . -type f -name foo\*
### delete
find . -type f -name foo\* -delete

## interactive session
### Galvani a100
srun --job-name "InteractiveJob" --partition=a100-galvani --ntasks=1 --nodes=1 --gres=gpu:2 --time 1:00:00 --pty bash

## get architecture/driver info 
nvidia-smi

## tmux
tmux ls <!-- list running jobs -->
tmux new -s SESSION_NAME <!-- create job -->
... <!-- run job (inside tmux terminal) -->
CTRL+B+D <!-- detach job (can also close the terminal via vsc) -->
tmux attach -t SESSION_NAME <!-- attach job -->
exit <!-- exit job (inside tmux terminal) -->
tmux kill-session -t SESSION_NAME <!-- terminate session -->

## gpustat
gpustat -a <!-- discrete -->
gpustat -i <!-- continuous -->

## download huggingface model
huggingface-cli download Qwen/Qwen2.5-3B --local-dir ../../qwen2.5_3b