#!/bin/bash
#SBATCH --job-name=BaLu_GNN_Syn_M=None_SimRel=1_Rel=4_MCAR_BaLu_GRAPE
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
cd /mnt/vast-kisski/projects/kisski-tib-activecl/BaLu_Plus/
echo "--- Running: run_exps.py ---"
python3 -u run_exps.py --model_name BaLu_Plus --imputer BaLu_GRAPE --dataset Syn_M=None_SimRel=1_Rel=4_MCAR --missing_p 0.0  --gconv GCN --rconv GCN --imputer_node_dims 64 64 --interference_node_dims 64
python3 -u run_exps.py --model_name BaLu_Plus --imputer BaLu_GRAPE --dataset Syn_M=None_SimRel=1_Rel=4_MCAR --missing_p 0.1  --gconv GCN --rconv GCN --imputer_node_dims 64 64 --interference_node_dims 64
python3 -u run_exps.py --model_name BaLu_Plus --imputer BaLu_GRAPE --dataset Syn_M=None_SimRel=1_Rel=4_MCAR --missing_p 0.3  --gconv GCN --rconv GCN --imputer_node_dims 64 64 --interference_node_dims 64

echo "===== JOB COMPLETED ====="
echo "Date: $(date)"
