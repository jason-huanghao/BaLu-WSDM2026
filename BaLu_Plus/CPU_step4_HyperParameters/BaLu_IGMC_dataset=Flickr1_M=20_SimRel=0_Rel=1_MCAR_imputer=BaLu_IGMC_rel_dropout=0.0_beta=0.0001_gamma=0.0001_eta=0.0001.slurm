#!/bin/bash
#SBATCH --job-name=BaLu_GNN_Flickr1_M=20_SimRel=0_Rel=1_MCAR_BaLu_IGMC
#SBATCH --partition=jupyter:cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=run_%x_%j.out
#SBATCH --error=run_%x_%j.err
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
# module load gcc/13.2.0
# module load cuda/12.6.2
# Set up conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/vast-kisski/projects/kisski-tib-activecl/cenv
# Diagnostics
echo "Which python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
# Navigate to project directory
cd /mnt/vast-kisski/projects/kisski-tib-activecl/BaLu_Plus/
echo "--- Running: run_exps.py ---"
python3 -u run_exps.py --model_name BaLu_Plus --imputer BaLu_IGMC --dataset Flickr1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.0  --gconv GCN --rconv GCN --rel_dropout 0.0 --beta 0.0001 --gamma 0.0001 --eta 0.0001
python3 -u run_exps.py --model_name BaLu_Plus --imputer BaLu_IGMC --dataset Flickr1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.1  --gconv GCN --rconv GCN --rel_dropout 0.0 --beta 0.0001 --gamma 0.0001 --eta 0.0001
python3 -u run_exps.py --model_name BaLu_Plus --imputer BaLu_IGMC --dataset Flickr1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.3  --gconv GCN --rconv GCN --rel_dropout 0.0 --beta 0.0001 --gamma 0.0001 --eta 0.0001

echo "===== JOB COMPLETED ====="
echo "Date: $(date)"
