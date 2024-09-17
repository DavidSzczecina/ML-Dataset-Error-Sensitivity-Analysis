#!/bin/bash
#SBATCH --job-name=bioscan_cleaning
#SBATCH --account=def-pfieguth
#SBATCH --time=7:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL

# Run virtual environment
source ../../envs/env3/bin/activate
module load python
cd projects/def-pfieguth/dszczeci/BIOSCAN-1M-CleanLab/BIOSCAN-1M-main
python dataCleaning.py --save_model --update_model --save_model_path largeDatasetTrain --num_folds 5 --epochs 1 --order large_insect_order --order_split train
