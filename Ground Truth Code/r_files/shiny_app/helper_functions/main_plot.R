library(readr)
library(dplyr)
library(ggplot2)
source('../functions/search_dir.r')
source('helper_functions/get_matching_data.R')
source('helper_functions/name_change.R')
#graphs_data_vect is the list of lists, with each index being the list of csv files for each graph line
#Also need to pass in the loss checkbox matrix (list containing the checkbox vectors for each graph line)
main_plot <- function(graphs_data_vect, data_amount, metric, num_graphs, num_epochs_vect, loss_checkbox_matrix){
    #General graph setup
    metric_label <- name_change(metric)
    graph <- ggplot() +
    labs(x = "Corruption Rate", y = metric_label) +
    ggtitle(paste0(metric_label, " vs Corruption Rate")) +
    theme(aspect.ratio = 3/4,  panel.grid.major = element_line(colour = "gray", size = 0.5),
    # panel.grid.minor = element_line(colour = "black", size = 0.25), 
      panel.background = element_rect(fill = "white", colour = NA),
      legend.position = c(0.2, 0.5), 
      legend.box.background = element_rect(color = "black", size = 1),
      legend.key = element_rect(color = "black", fill = "white"),
      legend.title = element_blank(), 
      plot.title = element_text(hjust = 0.5, size = 18)) + 
      scale_x_continuous(breaks = seq(0, 1, by=0.1))+
    
    if(metric == "accuracy" | metric == "avg_sens" | metric == "avg_spec"){
        scale_y_continuous(
            limits = c(0, 100), 
            breaks = seq(from = 0, to = 100, by = 10)
        )
    }
    else if(metric == "loss"){
        # scale_y_continuous(breaks = seq(from = floor(min()), 
        # to = 6, 
        # by = 0.25))
    }
    #Check if the user wants to plot a single job or summary stats
    if(data_amount == "single"){

        for(i in 1:num_graphs){
            dataframe_single <- graphs_data_vect[[i]][[length(graphs_data_vect[[i]])]]
            final_epoch_data <- dataframe_single %>%
            filter(epoch == num_epochs_vect[i]) %>%
            arrange(corruption_rate)
    
            if(!any(is.na(loss_checkbox_matrix[[i]]))){
                for(j in seq_along(loss_checkbox_matrix[[i]])){
                y_col <- loss_checkbox_matrix[[i]][[j]]
                y_col_aesthetic <- name_change(y_col)
                colour_label <- paste0("Graph ", i, " ", y_col_aesthetic)
                colour_label_quoted <- shQuote(colour_label, type = "sh")
                graph <- graph + 
                geom_line(data = final_epoch_data, aes_string(x = "corruption_rate", y = y_col, colour = colour_label_quoted), size = 1) +
                geom_point(data = final_epoch_data, aes_string(x = "corruption_rate", y = y_col, colour = colour_label_quoted), size = 2)
                }
            } 
            else {
                y_col <- metric
                colour_label <- paste0("Graph ", i)
                colour_label_quoted <- shQuote(colour_label, type = "sh")
                graph <- graph + 
                geom_line(data = final_epoch_data, aes_string(x = "corruption_rate", y = y_col, colour = colour_label_quoted), size = 1) +
                geom_point(data = final_epoch_data, aes_string(x = "corruption_rate", y = y_col, colour = colour_label_quoted), size = 2)
            }
        }

        return(graph)
    }
  
    else{
        #Loop through # of graph lines
        for(i in 1:num_graphs){
            #Find most up to date folder name
            latest_stats_file <- names(graphs_data_vect[[i]])[length(graphs_data_vect[[i]])]
            #Check for loss checkboxes
            if(!any(is.na(loss_checkbox_matrix[[i]]))){
                #Loop through the loss checkboxes vector for each graph line
                for(j in seq_along(loss_checkbox_matrix[[i]])){
                    #Get the loss type
                    loss_type <- loss_checkbox_matrix[[i]][[j]]
                    #Change the name to it's proper cap aesthetic 
                    y_col_aesthetic <- name_change(loss_type)
                    #Summary stats dataframe from the latest job
                    summary_stats_df <- search_directory_csv(paste0('../../job_outputs/', latest_stats_file), paste0("summary_stats_", loss_type, ".csv"))
                    #Get the mean , corruption, and stdev vectors from the summary stats dataframe
                    mean_vect <- safe_extract_csv(summary_stats_df, "mean")
                    corruption_vect <- safe_extract_csv(summary_stats_df, "corruption_rate")
                    stdev_vect <- safe_extract_csv(summary_stats_df, "stdev")
                    #Colour labels 
                    colour_label <- paste0("Graph ", i, " ", y_col_aesthetic)
                    #colour_label_quoted <- shQuote(colour_label, type = "sh")
                    #Upper and lower data for the ribbon plot
                    upper_data <- mean_vect + stdev_vect
                    lower_data <- mean_vect - stdev_vect
                    ribbon_data <- data.frame(corruption_rate = corruption_vect, 
                          mean = mean_vect, 
                          stdev = stdev_vect, 
                          upper = upper_data, 
                          lower = lower_data, 
                          colour_label = factor(colour_label))
                    #Add the ribbon plot to the graph
                    graph <- graph + 
                    geom_line(data = ribbon_data, aes(x = corruption_rate, y = mean, colour = colour_label), size = 1) +
                    geom_ribbon(data = ribbon_data, aes(x = corruption_rate, ymin = lower, ymax = upper, fill = colour_label), alpha = 0.2) +
                    geom_point(data = ribbon_data, aes(x = corruption_rate, y = mean, colour = colour_label), size = 2)
                
                }
            } 
            else{
               summary_stats_df <- search_directory_csv(paste0('../../job_outputs/', latest_stats_file), paste0("summary_stats_", metric, ".csv"))

                mean_vect <- safe_extract_csv(summary_stats_df, "mean")
                corruption_vect <- safe_extract_csv(summary_stats_df, "corruption_rate")
                stdev_vect <- safe_extract_csv(summary_stats_df, "stdev")
                colour_label <- paste0("Graph ", i)
                upper_data <- mean_vect + stdev_vect
                lower_data <- mean_vect - stdev_vect
                ribbon_data <- data.frame(corruption_rate = corruption_vect, 
                          mean = mean_vect, 
                          stdev = stdev_vect, 
                          upper = upper_data, 
                          lower = lower_data, 
                          colour_label = factor(colour_label))
                
                #Add the ribbon plot to the graph
                graph <- graph + 
                    geom_line(data = ribbon_data, aes(x = corruption_rate, y = mean, colour = colour_label), size = 1) +
                    geom_ribbon(data = ribbon_data, aes(x = corruption_rate, ymin = lower, ymax = upper, fill = colour_label), alpha = 0.2) +
                    geom_point(data = ribbon_data, aes(x = corruption_rate, y = mean, colour = colour_label), size = 2)
                
            }
           
        }
         return(graph)
    }
}   