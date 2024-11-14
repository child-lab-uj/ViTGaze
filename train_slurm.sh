#!/bin/bash -l

## Job Name
#SBATCH -J child-lab-vitgaze
## Number of allocated nodes
#SBATCH -N 1
## Number of tasks per node (default is the number of allocated cores per node)
#SBATCH --ntasks-per-node=1
## Amount of memory per CPU core (default is 5GB per core)
#SBATCH --mem-per-cpu=1GB
## Maximum job duration (format HH:MM:SS)
#SBATCH --time=01:00:00 
## Grant name for resource usage accounting
#SBATCH -A plgrobot
## Partition specification
#SBATCH -p plgrid-testing
## Standard output file
#SBATCH --output="output.out"
## Standard error file
#SBATCH --error="error.err"

## Switching to the directory where the sbatch command was initiated
cd $SLURM_SUBMIT_DIR

conda create -n ViTGaze python==3.9.18
conda activate ViTGaze

pip install -r requirements.txt

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

export CUDA_VISIBLE_DEVICES="0"

config_files=(
    # "configs/gazefollow.py"
    # "configs/gazefollow_518.py"
    "configs/videoattentiontarget.py"
)

run_experiment() {
    local config="$1"
    echo "Running experiment with config: $config"
    python -u tools/train.py --config-file "$config" --num-gpu 2
}

for config in "${config_files[@]}"
do
    run_experiment "$config" &
    pid=$!
    wait "$pid"
    sleep 10
done
