library(readr)
library(dplyr)

source("../functions/search_dir.r")
source('../functions/safe_extract.r')

get_valid_data <- function(){
    # Get all the folders in the job_outputs directory
    all_folders <- list.dirs("../../job_outputs", full.names = FALSE, recursive = FALSE)
    # Filter for the folders of job outputs 
    wanted_folders <- grep("files_", all_folders, value = TRUE)
    #Initialize empty list of valid folders
    valid_folders <- list()
    #Loop through the folders, grabbing the metadata and checking if it's valid
    for(name in wanted_folders){
         metadata <- search_directory_json(paste0("../../job_outputs/",name), "job_details.json")
         if(is.null(metadata)){
            next
         }
         temp_corruption_vect <- safe_extract_json(metadata, "corruption_arr")
         temp_epochs <- safe_extract_json(metadata, "num_epochs")
         temp_size <- safe_extract_json(metadata, "data_chunk")
         temp_num_classes <- safe_extract_json(metadata, "num_classes")
         temp_datset_type <- safe_extract_json(metadata, "dataset")
         temp_model_architecture <- safe_extract_json(metadata, "model_name")
         temp_version <- safe_extract_json(metadata, "version")
         if(!is.null(temp_corruption_vect) & !is.null(temp_epochs) & !is.null(temp_size) & !is.null(temp_num_classes)
            & !is.null(temp_datset_type) & !is.null(temp_model_architecture) & !is.null(temp_version)){
            valid_folders[[name]] <- metadata
         }
    }
    
    #Return the named list of valid folders, with a file name -> metadata_df mapping
    return(valid_folders)
}