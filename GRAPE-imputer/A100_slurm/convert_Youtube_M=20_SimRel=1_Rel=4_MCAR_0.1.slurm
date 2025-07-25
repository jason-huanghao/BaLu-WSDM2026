#!/bin/bash
#SBATCH --job-name=Youtube_M=20_SimRel=1_Rel=4_MCAR
#SBATCH -p kisski
#SBATCH -G A100:1                    
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=convert_Youtube_M=20_SimRel=1_Rel=4_MCAR_%j.out
#SBATCH --error=convert_Youtube_M=20_SimRel=1_Rel=4_MCAR_%j.err
#SBATCH --mail-type=FAIL # Email on start, end, failure
#SBATCH --mail-user=hao.huang@tib.eu # <-- Replace with your real email
echo "===== JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "User: $USER"
# Load environment
module load miniforge3
# module load cuda/11.8
module load gcc/13.2.0
module load cuda/11.8

# Set up conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/vast-kisski/projects/kisski-tib-activecl/cenv
# Diagnostics
echo "Which python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
# Navigate to project directory
cd /mnt/vast-kisski/projects/kisski-tib-activecl/GRAPE-imputer
# Run scripts
echo "--- Running: run_data_prepare.py"

python -u run_grape.py --dataset Youtube_M=20_SimRel=1_Rel=4_MCAR --missing_p 0.1 

echo "===== JOB COMPLETED ====="
echo "Date: $(date)"
