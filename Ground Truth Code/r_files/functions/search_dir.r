library(jsonlite)


search_directory_json <- function(base_dir, file_name){
    found_files <- list.files(path = base_dir, pattern = paste0("^", file_name, "$"), recursive = TRUE, full.names = TRUE)
    if (length(found_files) > 0) {
    # If found, read the first occurrence (or handle duplicates as needed)
    data <- fromJSON(found_files[length(found_files)])
    return(data)
    } else {
    return(NULL)
    }
}

search_directory_csv <- function(base_dir, file_name){
    found_files <- list.files(path = base_dir, pattern = paste0("^", file_name, "$"), recursive = TRUE, full.names = TRUE)
    if (length(found_files) > 0) {
    # If found, read the first occurrence (or handle duplicates as needed)
    data <- read.csv(found_files[length(found_files)])
    return(data)
    } else {
    return(NULL)
    }
}



