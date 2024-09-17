# Replacement for optional chaining found in JavaScript. Access column of dataframe if it exists, or return NULL. 
safe_extract_json <- function(my_list, key) {
  # Check if the argument is a list and the key exists in the list
  if (is.list(my_list) && key %in% names(my_list)) {
    return(my_list[[key]])
  }
  # Return NULL if the key is not found
  return(NULL)
}


safe_extract_csv <- function(df, colname) {
  if (is.data.frame(df) && colname %in% names(df)) {
    return(df[[colname]])
  }
  return(NULL)
}