# Ground-truth-error-sensitivity-analysis
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

