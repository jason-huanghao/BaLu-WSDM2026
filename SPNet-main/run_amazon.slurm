#!/bin/bash
#SBATCH --job-name=SPNet
#SBATCH --partition=kisski
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=run_%x_%j.out
#SBATCH --error=run_%x_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL          # Email on start, end, failure
#SBATCH --mail-user=hao.huang@tib.eu   # <-- Replace with your real email


echo "===== JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Date:     $(date)"
echo "User:     $USER"


# Load environment
module load miniforge3
module load cuda/11.8

# Set up conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/vast-kisski/projects/kisski-tib-activecl/cenv

# Diagnostics
echo "Which python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Navigate to project directory
cd ~/SPNet-main_hao || { echo "❌ Failed to cd into ~/SPNet-main_hao"; exit 1; }

# Set the environment variable for the job
export CUDA_LAUNCH_BLOCKING=1

# Run scripts
echo "--- Running: main_compare_balu.py Amazon ---"
python3 -u main_compare_balu.py --dataset AMZS --missing_p 0.0
python3 -u main_compare_balu.py --dataset AMZS --missing_p 0.1
python3 -u main_compare_balu.py --dataset AMZS --missing_p 0.3
python3 -u main_compare_balu.py --dataset AMZS --missing_p 0.5

echo "===== JOB COMPLETED ====="
echo "Date: $(date)"