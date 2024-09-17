#!/bin/bash
#SBATCH --job-name=gt_sensitivity_analysis
#SBATCH --account=def-pfieguth
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=2   # CPU cores/threads
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err

# Load needed python and cuda modules
module load python/3.10.2 cuda cudnn
module load scipy-stack/2023b
module load r/4.3.1
export R_LIBS=/home/$USER/projects/rrg-pfieguth/n37zhao/R/4.3.1
# Run virtual environment
source ../../envs/env1/bin/activate

#Create a new directory to store the results of the sensitivity analysis

base_folder="../job_outputs/files_$SLURM_JOBID"
model_params_folder="../job_outputs/files_$SLURM_JOBID/model_params" 
csv_folder="../job_outputs/files_$SLURM_JOBID/csv_data"
txt_folder="../job_outputs/files_$SLURM_JOBID/txt_data"
#Value for which to divide training data by 
min_size=1

#Create the new folders
mkdir -p $csv_folder
mkdir -p $model_params_folder
mkdir -p $txt_folder

CORRUPTION_RATES=()
#List the desired corruption rates (Don't change this!!! needs to be consistent to collect summary stats)
#This one is for 10 classes
CORRUPTION_RATES_10=(0 0.25 0.5 0.6 0.7 0.75 0.775 0.8 0.825 0.85 0.8625 0.875 0.8875 0.9 0.9125 0.925 0.95 0.975 1)
#This one is for 6 classes
CORRUPTION_RATES_6=(0 0.25 0.5 0.6 0.7 0.75 0.775 0.7875 0.8 0.8125 0.825 0.8375 0.85 0.8625 0.875 0.8875 0.9 0.95 1)
#This one is for 2 classes 
CORRUPTION_RATES_2=(0 0.25 0.35 0.4 0.425 0.4375 0.45 0.4625 0.475 0.4875 0.5 0.5125 0.525 0.5375 0.55 0.5625 0.575 0.6 0.65 0.7 0.75 0.85 0.9 0.95 0.975 1)


#Set the number of epochs to compute 
num_epochs=2

# Set the classes to be used for the sensitivity analysis
num_classes=-1
#Create a csv file with the results of the sensitivity analysis
touch "$csv_folder/data.csv"
#Add the header to the csv file
echo "corruption_rate,epoch,accuracy,training_loss,test_loss,avg_sens,avg_spec" > $csv_folder/data.csv 

#Set the seed value for reproducibility
seed_value=1
dataset="MNIST"
model_architecture="CNN"



#echo "num_epochs,seed_value,min_size,corruption_rates,num_classes,dataset,model_architecture,version" > $csv_folder/metadata.csv

if [ $num_classes -eq 10 ]; then 
    CORRUPTION_RATES=("${CORRUPTION_RATES_10[@]}")
elif [ $num_classes -eq 6 ]; then 
    CORRUPTION_RATES=("${CORRUPTION_RATES_6[@]}")
elif [ $num_classes -eq 2 ]; then 
    CORRUPTION_RATES=("${CORRUPTION_RATES_2[@]}")
else 
    CORRUPTION_RATES=(0 0.25 0.5 0.6 0.7 0.75 0.775 0.8 0.825 0.85 0.8625 0.875 0.8875 0.9 0.9125 0.925 0.95 0.975 1)
fi
 
#CORRUPTION_RATES=(0 0.25 0.5)

isManual="true"
#Run the sensitivity analysis for each corruption rate
for RATE in "${CORRUPTION_RATES[@]}";
do
    echo "Running with corruption rate: $RATE"
    #Add metadata to the metadata file
    #Pass in corrruption rate and number of epoch parameters
    python3 ../python_files/prune_sens_analysis.py --corruption_rate "$RATE" --dataset "$dataset" --model_architecture "$model_architecture" --num_classes "$num_classes" --base_folder "$base_folder" --data_chunk "$min_size" --epochs "$num_epochs" --jobid "$SLURM_JOBID" --seed "$seed_value" --manual "$isManual" --all_corruption "${CORRUPTION_RATES[@]}" >> $txt_folder/network_output.txt 2>> $txt_folder/network_error.txt
done 

#Run the python script to plot the results and generate the PNG files 

python3 ../python_files/runCalcs.py --jobid "$SLURM_JOBID" >> $txt_folder/network_output.txt 2>> $txt_folder/network_error.txt

#Run the R script to plot the results and generate the PNG files
#Rscript ../r_files/plot_results.r --jobid "$SLURM_JOBID" --epochs "$num_epochs" --base_folder "$base_folder" --data_chunk "$min_size" --num_classes "$num_classes" 

deactivate 

