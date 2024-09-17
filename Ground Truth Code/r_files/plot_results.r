library(readr)
library(dplyr)
library(argparse)
library(ggplot2) 
library(ggrepel)
source("../r_files/functions/graph2d.r")
source("../r_files/functions/summary_stats.r")
source("../r_files/functions/boxplot.r")
source("../r_files/functions/ribbon_plot.r")
source("../r_files/functions/search_dir.r")
#Parse metadata arguments
parser <- ArgumentParser(description = "Parse data.csv")
parser$add_argument("--jobid", type = "integer", help = "jobid", required = TRUE)
parser$add_argument("--epochs", type = "integer", help = "epoch", required = TRUE)
parser$add_argument("--base_folder", type = "character", help = "base folder path", required = TRUE)
parser$add_argument("--data_chunk", type = "integer", help = "size of training data used", required = TRUE)
parser$add_argument("--num_classes", type = "integer", help = "num classees", default = 10, required = FALSE)
args <- parser$parse_args()

jobid_char <- as.character(args$jobid)

#Read in the data.csv file

corruption_data <- search_directory(args$base_folder, "data.csv")
#Filter for the final epoch
final_epoch_df <- corruption_data %>%
  filter(epoch == args$epochs) %>%
  arrange(corruption_rate)

#Pass this into get_data function, it is a dataframe with only the corruption rates
corruption_rate_df <-corruption_data %>%
  filter(epoch == args$epochs) %>%
  select(corruption_rate)

#Read in the metadata.csv file
metadata_df <- search_directory(args$base_folder, "metadata.csv")

#Corruption vector used in this job 
true_corruption_vect <- metadata_df$corruption_rates
#Seed value used in this job
true_seed <- metadata_df$seed_value[1]
true_model_arch <- metadata_df$model_architecture[1]

#Get the summary stats for these particular metrics across all jobs run with the same metadata
accuracy_df <- get_data(args$data_chunk, args$epochs, true_seed, true_corruption_vect, "accuracy", corruption_rate_df, args$base_folder, args$num_classes, true_model_arch)
avg_sens_df <- get_data(args$data_chunk, args$epochs, true_seed, true_corruption_vect, "avg_sens", corruption_rate_df, args$base_folder, args$num_classes, true_model_arch)
avg_spec_df <- get_data(args$data_chunk, args$epochs, true_seed, true_corruption_vect, "avg_spec", corruption_rate_df, args$base_folder, args$num_classes, true_model_arch)
test_loss_df <- get_data(args$data_chunk, args$epochs, true_seed, true_corruption_vect, "test_loss", corruption_rate_df, args$base_folder, args$num_classes, true_model_arch)
training_loss_df <- get_data(args$data_chunk, args$epochs, true_seed, true_corruption_vect, "training_loss", corruption_rate_df, args$base_folder, args$num_classes, true_model_arch)

#Create boxplots for each metric
boxplot_summary(accuracy_df, "accuracy", args$base_folder)
boxplot_summary(avg_sens_df, "avg_sens", args$base_folder)
boxplot_summary(avg_spec_df, "avg_spec", args$base_folder)
boxplot_summary(test_loss_df, "test_loss", args$base_folder)
boxplot_summary(training_loss_df, "training_loss", args$base_folder)

#Graph this specific job's metrics 
graph_data(args$base_folder, final_epoch_df, "accuracy")
graph_data(args$base_folder, final_epoch_df, "avg_sens")
graph_data(args$base_folder, final_epoch_df, "avg_spec")

#Get the dataframe for loss 
loss_df <- final_epoch_df %>%
  select(corruption_rate, training_loss, test_loss)

#Graph this specific job's losses
graph4 <- ggplot(loss_df, aes(x=corruption_rate)) +
geom_line(aes(y=training_loss, colour = "Training Loss")) +
geom_line(aes(y=test_loss, colour = "Test Loss")) +
geom_point(aes(y=training_loss, colour = "Training Loss")) +
geom_point(aes(y=test_loss, colour = "Test Loss")) + 
#geom_text_repel(aes(y = training_loss, label = training_loss, colour = "Training Loss"), box.padding = 0.5) +
#geom_text_repel(aes(y = test_loss, label = test_loss, colour = "Test Loss"), box.padding = 0.5) +
#ggtitle("Losses vs Corruption Rate") +
xlab("Corruption Rate") +
ylab("Loss") +
labs(colour = "Losses") +
theme(plot.title = element_text(hjust = 0.5), aspect.ratio = 3/4) +
scale_x_continuous(breaks = seq(0, 1, by=0.1))
ggsave(filename =  paste0(args$base_folder, "/graphics/loss.png"), plot = graph4)

#Ribbot plot for each metric
ribbon_plot("accuracy", args$base_folder)
ribbon_plot("avg_sens", args$base_folder)
ribbon_plot("avg_spec", args$base_folder)
ribbon_plot("test_loss", args$base_folder)
ribbon_plot("training_loss", args$base_folder)