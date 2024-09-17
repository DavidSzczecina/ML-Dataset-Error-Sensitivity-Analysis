# ML-Dataset-Error-Sensitivity-Analysis
Conducting resarch on Error Sensitivity and Dataset Error Pruning for Machine Learning datasets. 
Done under the supervision of Profesor Nicholas Pellegrino at the Waterloo Vision and Image Processing Lab.  


## Overview
All of the code in this repository was used to increase accuracy and provide more insight to the BIOSCAN-1M and BIOSCAN-5M datasets. Through the use of dataset label error detection, mislabeled data was identified and pruned to provide substantial accuracy improvments for certain classes within the dataset.


## Contents



### Dataset_Error_Detection_and_Pruning
Performed analysis of how accuracy and loss are affected by mislabeling of common ML datasets
Models are used to predict on their own data by using a probabilistic approach to find noisy labels in datasets. 

It works by analyzing model predictions and comparing them to the given labels to identify potential label errors or ambiguities. 

To prevent the bias of predicting on data a model trained on, we use cross fold validation to break the dataset into n folds, training on n-1 folds and predicting on the left out portion to achive out of sample probabilities. With these predicted probabilities we can then analize using a confusion matrix which data il likely to be mislabeled.  



### Ground-truth-error-sensitivity-analysis
Analysis code to test the sensitivity of various network models to errors in ground truth training data. 
A Project for the University of Waterloo.

Structure: 
Jobs are queued in the gt_sens.sh file. Here, all metadata pertaining to the network is defined. This includes parameters like 
number of epochs, network architecture, etc. This bash scripts is also responsible for creating the job_output folder, with each folder containing the unique job_id for each run. Inside the job_output folder all pertinent information can be found. The python script called will execute the computations of the network. This python script outputs a data.csv file which contains statistics like accuracy, loss, sensitivity etc. The shiny app can then be started, using the bash script. This app will have fields in which the user can change to display and graph different data. It searches through the job_outputs folder for a matching folder, using the data.csv inside of it to graph. This is done for each graph-line that is desired. To run the Shiny app, enter the "slurm_scripts" directory, then run the "shinyApp.sh" bash file. 

Startup: 
Activate virtual env: "source {directory to environment bin}/activate
Plot GUI: cd into "slurm_scripts" directory, then run the bash script w/ "./shinyApp.sh" 
Run a job: cd into "slurm_scripts" directory, then run the Bash script w/ "sbatch gt_sens.sh" 
Run unit tests: (ensure in virtual env) cd into "python_files" directory, then run "python tests/fileUtilsTest.py



### Error Sensitivity and Pruning
Project where we combine error sensitivity analysis with label error pruning to evaluate the effectiveness of error detection methods



### BIOSCAN-1M Insect
All of the code in this repository was used to increase accuracy and provide more insight to the BIOSCAN-1M and BIOSCAN-5M datasets. Through the use of dataset label error detection, mislabeled data was identified and pruned to provide substantial accuracy improvments for certain classes within the dataset.

###### <h3> Overview
This repository houses the codes and data pertaining to the [BIOSCAN-1M-Insect project](https://biodiversitygenomics.net/1M_insects/). 
Within this project, we introduce the __BIOSCAN-1M Insect dataset__, which can be accessed 
for download via the provided links. The repository encompasses code for data sampling and splitting, 
dataset statistics analysis, as well as image-based classification experiments centered around 
the taxonomy classification of insects. 


Anyone interested in using BIOSCAN-1M Insect dataset and/or the corresponding code repository, please cite the [Paper](http://arxiv.org/abs/2307.10455):