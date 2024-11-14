#!/bin/bash -l

## Job Name
#SBATCH -J child-lab-vitgaze
## Number of allocated nodes
#SBATCH -N 1
## Number of tasks per node (default is the number of allocated cores per node)
#SBATCH --ntasks-per-node=1
## Amount of memory per CPU core (default is 5GB per core)
#SBATCH --mem-per-cpu=5GB
## Maximum job duration (format HH:MM:SS)
#SBATCH --time=00:30:00 
## Grant name for resource usage accounting
#SBATCH --account=plgrobot-gpu-a100
## Partition specification
#SBATCH --partition=plgrid-gpu-a100
## Cpus:
#SBATCH --cpus-per-task=2
## Standard output file
#SBATCH --output="output.out"
## Standard error file
#SBATCH --error="error.err"
## GPU
#SBATCH --gres=gpu
## Switching to the directory where the sbatch command was initiated
cd $SLURM_SUBMIT_DIR

module load Miniconda3

eval "$(conda shell.bash hook)"

# module load python

# generate the head masks
conda create --prefix $SCRATCH/.conda/retinaface python==3.9.18 || true
conda activate $SCRATCH/.conda/retinaface

pip config --user set global.cache-dir $SCRATCH/.cache

pip install -U retinaface_pytorch
python scripts/gen_gazefollow_head_masks.py --dataset_dir $PLG_GROUPS_STORAGE/plggrai/jkosmydel/datasets/videoattentiontarget --subset train

# install the requirements
conda create --prefix $SCRATCH/.conda/ViTGaze python==3.9.18 || true
conda activate $SCRATCH/.conda/ViTGaze

pip config --user set global.cache-dir $SCRATCH/.cache

pip install -r requirements.txt

python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# create the dinov2 pretrained 
mkdir pretrained
wget -P pretrained https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
python scripts/convert_pth.py --src pretrained/dinov2_vits14_pretrain.pth --dst pretrained/dinov2_small.pth

export DATA_ROOT=$PLG_GROUPS_STORAGE/plggrai/jkosmydel/datasets/

export CUDA_VISIBLE_DEVICES="0"

config_files=(
    # "configs/gazefollow.py"
    # "configs/gazefollow_518.py"
    "configs/videoattentiontarget.py"
)

run_experiment() {
    local config="$1"
    echo "Running experiment with config: $config"
    srun python -u tools/train.py --config-file "$config" --num-gpu 2
}

for config in "${config_files[@]}"
do
    run_experiment "$config" &
    pid=$!
    wait "$pid"
    sleep 10
done
