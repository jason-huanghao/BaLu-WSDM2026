#!/bin/bash

# Submit all conversion jobs

sbatch A100_slurm/convert_Syn_M=None_SimRel=1_Rel=4_MCAR_0.1.slurm
sbatch A100_slurm/convert_Syn_M=None_SimRel=1_Rel=4_MCAR_0.3.slurm
sbatch A100_slurm/convert_Youtube_M=20_SimRel=1_Rel=4_MCAR_0.1.slurm
sbatch A100_slurm/convert_Youtube_M=20_SimRel=1_Rel=4_MCAR_0.3.slurm
sbatch A100_slurm/convert_BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR_0.1.slurm
sbatch A100_slurm/convert_BlogCatalog1_M=20_SimRel=0_Rel=1_MCAR_0.3.slurm
sbatch A100_slurm/convert_Flickr1_M=20_SimRel=0_Rel=1_MCAR_0.1.slurm
sbatch A100_slurm/convert_Flickr1_M=20_SimRel=0_Rel=1_MCAR_0.3.slurm
