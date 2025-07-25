#!/bin/bash

# Submit all conversion jobs

sbatch H100_slurm/convert_Syn_MCAR_0.0.slurm
sbatch H100_slurm/convert_Syn_MCAR_0.1.slurm
sbatch H100_slurm/convert_Syn_MCAR_0.3.slurm
sbatch H100_slurm/convert_BlogCatalog1_MCAR_0.0.slurm
sbatch H100_slurm/convert_BlogCatalog1_MCAR_0.1.slurm
sbatch H100_slurm/convert_BlogCatalog1_MCAR_0.3.slurm
sbatch H100_slurm/convert_Flickr1_MCAR_0.0.slurm
sbatch H100_slurm/convert_Flickr1_MCAR_0.1.slurm
sbatch H100_slurm/convert_Flickr1_MCAR_0.3.slurm
sbatch H100_slurm/convert_Youtube_MCAR_0.0.slurm
sbatch H100_slurm/convert_Youtube_MCAR_0.1.slurm
sbatch H100_slurm/convert_Youtube_MCAR_0.3.slurm
