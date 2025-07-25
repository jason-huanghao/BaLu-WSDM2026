#!/bin/bash
#SBATCH --job-name=GCN_HSIC_BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR
#SBATCH -p kisski
#SBATCH -G A100:1                    
#SBATCH --mem=20G
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
# Navigate to project directory
cd /mnt/vast-kisski/projects/kisski-tib-activecl/GCN_policy_ma 


# Run scripts
echo "--- Running: train_policy.py BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR ---"
python3 -u train_policy.py --dataset BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.0 --gnn_fun GCN
# python3 -u train_policy.py --dataset BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.1 --gnn_fun GCN
# python3 -u train_policy.py --dataset BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.3 --gnn_fun GCN


echo "===== JOB COMPLETED ====="
echo "Date: $(date)"
