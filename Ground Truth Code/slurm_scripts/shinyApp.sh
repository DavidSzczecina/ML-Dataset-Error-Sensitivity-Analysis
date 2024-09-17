#!/bin/bash

# Load the R module, adjust the version number as needed.
module load r/4.3.1

# Set the R_LIBS environment variable to the path where your R packages are located.
export R_LIBS=/home/$USER/projects/rrg-pfieguth/n37zhao/R/4.3.1
source ../../envs/env1/bin/activate
# Navigate to the directory containing your Shiny app. 
cd ../r_files/shiny_app

# Run the Shiny app using Rscript. Adjust the filename if your app file has a different name.
Rscript -e "shiny::runApp('app.R')"
