#!/bin/bash
#SBATCH --job-name=Flickr1_0.0
#SBATCH -p kisski-h100
#SBATCH -G H100:1                    
#SBATCH --mem=20G
#SBATCH --time=23:00:00
#SBATCH --output=convert_Flickr1_MCAR_%j.out
#SBATCH --error=convert_Flickr1_MCAR_%j.err
#SBATCH --mail-type=FAIL # Email on start, end, failure
#SBATCH --mail-user=hao.huang@tib.eu # <-- Replace with your real email
echo "===== JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $USER"
# Load environment
module load miniforge3

module load gcc/13.2.0
module load cuda/11.8

# Set up conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/vast-kisski/projects/kisski-tib-activecl/cenv
# Diagnostics
echo "Which python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
# Navigate to project directory
cd /mnt/vast-kisski/home/hao.huang01/u15611/Data-BaLu 
# Run scripts
echo "--- Running: run_data_prepare.py"

# python -u run_data_prepare.py --dataset Flickr1 --missing_p 0.1 --imputer 'ori_grape' --n_attr 10 --relation_sim 1 --n_rel 4
# python -u run_data_prepare.py --dataset Flickr1 --missing_p 0.3 --imputer 'ori_grape' --n_attr 10 --relation_sim 1 --n_rel 4

# python -u run_data_prepare.py --dataset Flickr1 --miss_pattern MCAR --missing_p 0.0 --n_attr 10 --relation_sim 0 --n_rel 1
# python -u run_data_prepare.py --dataset Flickr1 --miss_pattern MCAR --missing_p 0.1 --n_attr 10 --relation_sim 0 --n_rel 1

python -u run_data_prepare.py --dataset Flickr1 --miss_pattern MCAR --missing_p 0.0 --n_attr 10 --relation_sim 0 --n_rel 1

echo "===== JOB COMPLETED ====="
echo "Date: $(date)"
