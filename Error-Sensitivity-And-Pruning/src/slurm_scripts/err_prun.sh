#!/bin/bash
#SBATCH --job-name=err_prun_analysis
#SBATCH --account=def-pfieguth
#SBATCH --time=00:09:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2   # CPU cores/threads
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err


module load python/3.10.2 cuda cudann 
module load scipy-stack/2023b
module load r/4.3.1 
export R_LIBS=/home/$USER/projects/rrg-pfieguth/n37zhao/R/4.3.1
source ../../../envs/env1/bin/activate



base_folder="../../job_outputs/files_$SLURM_JOBID"
model_params_folder="../../job_outputs/files_$SLURM_JOBID/model_params" 
csv_folder="../../job_outputs/files_$SLURM_JOBID/csv_data"
txt_folder="../../job_outputs/files_$SLURM_JOBID/txt_data"

mkdir -p $csv_folder
mkdir -p $model_params_folder
mkdir -p $txt_folder