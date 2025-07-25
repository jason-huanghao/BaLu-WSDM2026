#!/bin/bash
#SBATCH --job-name=rl_BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR
#SBATCH -p jupyter
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=23:00:00
#SBATCH --output=run_%x_%j.out
#SBATCH --error=run_%x_%j.err

#SBATCH --mail-type=FAIL          # Email on start, end, failure
#SBATCH --mail-user=hao.huang@tib.eu   # <-- Replace with your real email


echo "===== JOB STARTED ====="
echo "Hostname: $(hostname)"
echo "Date:     $(date)"
echo "User:     $USER"


# Load environment
module load miniforge3
module load gcc/13.2.0
module load cuda/11.8.0
# module load gcc/13.2.0
# module load cuda/12.6.2

# Set up conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/vast-kisski/projects/kisski-tib-activecl/cenv

# Diagnostics
echo "Which python: $(which python)"
python -c "import torch; print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Navigate to project directory
cd /mnt/vast-kisski/projects/kisski-tib-activecl/OtherMethods

# Run scripts
echo "--- Running: main_other_methods.py BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR ---"

python3 -u main_other_methods.py --dataset BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.0 --em rl
python3 -u main_other_methods.py --dataset BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.1 --em rl
python3 -u main_other_methods.py --dataset BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR --missing_p 0.3 --em rl


echo "===== JOB COMPLETED ====="
echo "Date: $(date)"
