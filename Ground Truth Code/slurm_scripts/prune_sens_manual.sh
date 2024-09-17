#!/bin/bash

module load StdEnv/2020

# Load needed python and cuda modules
module load python/3.10.2 cuda cudnn
module load scipy-stack/2023b
module load r/4.3.1

echo module loadeed

dummy_id=999999
export R_LIBS=/home/$USER/projects/rrg-pfieguth/n37zhao/R/4.3.1
# Run virtual environment
source ../../envs/env1/bin/activate
base_folder="../job_outputs/files_$dummy_id"
model_params_folder="../job_outputs/files_$dummy_id/model_params" 
csv_folder="../job_outputs/files_$dummy_id/csv_data"
txt_folder="../job_outputs/files_$dummy_id/txt_data"
#Value for which to divide training data by 

#Create the new folders
mkdir -p $csv_folder
mkdir -p $model_params_folder
mkdir -p $txt_folder


SEED=1
CORRUPTION_RATES=(0 0.25 0.5 0.6 0.7 0.75 0.775 0.8 0.825 0.85 0.8625 0.875 0.8875 0.9 0.9125 0.925 0.95 0.975 1)

model_architecture='MLP'
dataset="MNIST"
num_classes=-1
epochs=12

echo "corruption_rate,epoch,accuracy,training_loss,test_loss,avg_sens,avg_spec" > $csv_folder/data.csv 

for rate in "${CORRUPTION_RATES[@]}"; 
do
    python3 ../python_files/prune_sens_analysis.py --dataset "$dataset" --seed "$SEED" --corruption_rate $rate --all_corruption "${CORRUPTION_RATES[@]}" --base_folder "$base_folder" --model_architecture "$model_architecture" --epochs "$epochs" --jobid "$dummy_id" --num_classes "$num_classes" 
done

