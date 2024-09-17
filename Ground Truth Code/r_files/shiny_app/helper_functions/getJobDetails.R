library(readr)
library(dplyr)




getJobDetails <- function(fileName){
    job_details <- search_directory_json(paste0("../../job_outputs/",fileName), "job_details.json")
    if(is.null(job_details)){
        stop("No job_details.json found")
    }
    criterion <- safe_extract_json(job_details, "criterion")
    scheduler <- safe_extract_json(job_details, "scheduler")
    optimizer <- safe_extract_json(job_details, "optimizer")
    hyperparams <- safe_extract_json(job_details, "hyperparams")
    return(list(hyperparams = hyperparams,optimizer = optimizer,scheduler =  scheduler,criterion =  criterion))
}
