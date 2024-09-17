library(readr)
library(dplyr)

source("../functions/search_dir.r")
source('../functions/safe_extract.r')

#This function returns a name list with a job_name -> csv_file mapping
#Valid folders is a named list, with a file name -> metadata_df mapping, csv_name is the csv in the job file to be returned 
#Rest of params are the user-chosen metadata
get_matching_data <- function(valid_folders, csv_name, num_classes, epochs, training_size, model_arch, dataset_type, version_type){

    #Initialize empty list of matching data, 
    matching_data <- list()
    #Loop through the valid folders, and check if the metadata matches the user-chosen metadata
    for(i in seq_along(valid_folders)){
        #Get the name of the folder
        key <- names(valid_folders)[i]
        #Get the metadata object
        metadata <- valid_folders[[i]]
        #Get the metadata from the df
        temp_epochs <- safe_extract_json(metadata, "num_epochs")
        temp_size <- safe_extract_json(metadata, "data_chunk")
        temp_num_classes <- safe_extract_json(metadata, "num_classes")
        temp_model_arch <- safe_extract_json(metadata, "model_name")
        temp_dataset_type <- safe_extract_json(metadata, "dataset")
        temp_version <- safe_extract_json(metadata, "version")
        #If the metadata matches, add it to the list of matching data
        if(!is.null(temp_epochs) & !is.null(temp_size) & !is.null(temp_num_classes) & !is.null(temp_model_arch) & !is.null(temp_dataset_type) & !is.null(temp_version)){
            if(version_type == -1){
                if(temp_epochs >= epochs & temp_size == training_size & temp_num_classes == num_classes 
                & temp_model_arch == model_arch & temp_dataset_type == dataset_type){
                    #Depending on the csv name, get the corresponding csv file 
                    csv_file <- search_directory_csv(paste0("../../job_outputs/", key), csv_name)
                    if(!is.null(csv_file)){
                        matching_data[[key]] <- csv_file
                    }
                } 
            }
            else{
                if(temp_epochs >= epochs & temp_size == training_size & temp_num_classes == num_classes 
                & temp_model_arch == model_arch & temp_dataset_type == dataset_type & temp_version == version_type){
                    #Depending on the csv name, get the corresponding csv file 
                    csv_file <- search_directory_csv(paste0("../../job_outputs/", key), csv_name)
                    if(!is.null(csv_file)){
                        matching_data[[key]] <- csv_file
                    }
                }
            }
        }
    }
    #This return will be a named list, with a job_name -> csv_file mapping
    return(matching_data)
}