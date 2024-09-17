#!/bin/bash
#SBATCH --job-name=gt_sensitivity_analysis
#SBATCH --account=def-pfieguth
#SBATCH --time=00:55:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2   # CPU cores/threads
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err

module load StdEnv/2020

# Load needed python and cuda modules
module load python/3.10.2 cuda cudnn
module load scipy-stack/2023b
module load r/4.3.1

export R_LIBS=/home/$USER/projects/rrg-pfieguth/n37zhao/R/4.3.1
# Run virtual environment
source ../../envs/env1/bin/activate
base_folder="../job_outputs/files_$SLURM_JOBID"
model_params_folder="../job_outputs/files_$SLURM_JOBID/model_params" 
csv_folder="../job_outputs/files_$SLURM_JOBID/csv_data"
txt_folder="../job_outputs/files_$SLURM_JOBID/txt_data"
#Value for which to divide training data by 

#Create the new folders
mkdir -p $csv_folder
mkdir -p $model_params_folder
mkdir -p $txt_folder


CORRUPTION_RATES=(0 0.25 0.5 0.6 0.7 0.75 0.775 0.8 0.825 0.85 0.8625 0.875 0.8875 0.9 0.9125 0.925 0.95 0.975 1)

echo "corruption_rate,epoch,accuracy,training_loss,test_loss,avg_sens,avg_spec" > $csv_folder/data.csv 

SEED=1

model_architecture='CNN'
dataset="MNIST"
num_classes=-1
epochs=2

echo Complete


for rate in "${CORRUPTION_RATES[@]}"; 
do    
    python3 ../python_files/prune_sens_analysis.py --dataset "$dataset" --seed "$SEED" --corruption_rate "$rate" --base_folder "$base_folder" --all_corruption "${CORRUPTION_RATES[@]}" --model_architecture "$model_architecture" --epochs "$epochs" --jobid "$SLURM_JOBID" --num_classes "$num_classes" >> $txt_folder/network_output.txt 2>> $txt_folder/network_error.txt
done




