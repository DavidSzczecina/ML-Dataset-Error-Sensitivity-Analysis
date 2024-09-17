library(shiny)
source('helper_functions/get_valid_data.R')
source('helper_functions/get_matching_data.R')
source('helper_functions/main_plot.R')
source('helper_functions/name_change.R')
source('helper_functions/getJobDetails.R')
source('helper_functions/hyperparamsFormat.R')
valid_data <- get_valid_data()



ui <- fluidPage(
    titlePanel("Ground Truth Error Sensitivity Analysis"),


    sidebarLayout(
        #Sidebar for selecting the graph to be generated
        sidebarPanel(
            #Choose between the summary stats or single job 
            radioButtons("data_amount", 
            label = "Use single job data, or aggregate for summary statistics", 
            choices = c("Independent" = "single", "Multiple" = "multi"),
            selected = "single", 
            inline = TRUE),
            
            #Choose the y-axis metric to be measured 
            selectInput("metric", 
            label = "Select the y-axis metric to analyze",
            choices = c("Accuracy" = "accuracy", "Loss" = "loss", "Sensitivity" = "avg_sens", "Specificity" = "avg_spec"), 
            multiple = FALSE),

            #Choose the number of different metadata parameters to compare 
            numericInput("num_graphs", 
            label = "Select number of graphs to place on this plot", 
            value =1,
            min = 1, 
            max = 4,
            step = 1),

            #Options based on above chosen value
            uiOutput("metadata_options")

        
        ),
        #The meat and potateos 
        mainPanel(
            tabsetPanel(
                #Tab for viewing Plots 
                tabPanel(strong("Plots"), 
                h1("Plot Section"),
                plotOutput("plot", width = '800px', height = '600px'),
                uiOutput("jobDetails")),
                #Tab for viewing raw data
                tabPanel(strong("Data"),
                h1("Data Section"),
                uiOutput("dataframes"))
            )
        )
   )

)


server <- function(input, output){
    
    bash_path <- "../../slurm_scripts/gt_sens_auto.sh"
    #Reactive value (boolean) to show the plot or not
    #Reactive value to store the number of graphs to be plotted
    output$metadata_options <- renderUI({
        #Initialize empty list of UI components
        ui_components <- list()

        #Depending on user input for number of graphs, change number of select options 
        for(i in 1:input$num_graphs){
            # List base on # of UI options 
            options_ui <- list(
                 # Header to differentiate each group of options 
                tags$h4(paste("Graph ", i)),

            # Select the number of epochs 
            numericInput(paste0("num_epochs", i), 
                         label = "Choose the number of epochs", 
                         value = 12, 
                         min = 1, 
                         max = 14),
                         

            # Select the number of classes
            numericInput(paste0("num_classes", i), 
                         label = "Choose the number of classes", 
                         value = 10, 
                         min = 2, 
                         max = 10),

            # Select the data chunk
            numericInput(paste0("data_chunk", i), 
                         label = "Choose data chunk for division", 
                         value = 1, 
                         min = 1, 
                         max = 8),

            selectInput(paste0("model_arch", i),
                        label = "Choose the model architecture",
                        choices = c("CNN" = "CNN", "MLP" = "MLP", "ResNet18" = "ResNet18", "VGG" = "VGG", "Vision Transformer" = "VIT", "Data Efficient Image Transformer" = "DEIT"),
                        multiple = FALSE, 
                        selected = "CNN"),

            selectInput(paste0("dataset_type",i),
                        label = "Choose the dataset",
                        choices = c("MNIST" = "MNIST", "FashionMNIST" = "FashionMNIST", "KMNIST" = "KMNIST", "CIFAR10"= "CIFAR10", "CIFAR100" = "CIFAR100"),
                        multiple = FALSE,
                        selected = "MNIST"),
                    
            numericInput(paste0("version", i),
                        label = "Choose model version",
                        value = 1,
            ),

           actionButton(paste0("job_run", i), "Run this job"),



            if(input$metric == "loss"){
                checkboxGroupInput(paste0("loss_type", i),
                                   label = "Choose the loss type(s) to display",
                                   choices = c("Training Loss" = "training_loss", "Test Loss" = "test_loss"),
                                   selected = "training_loss")
            }

          
            )

             # Append the group components to the list
            ui_components <- c(ui_components, options_ui)

        }
        #Taglist replaces need for div
        do.call(tagList, ui_components)
    })

    plot_viable <- reactiveVal(FALSE)
    err_msg <- reactiveVal("Error: Invalid inputs")
    num_epochs_vect_global <- reactiveVal()
    loss_checkbox_matrix_global <- reactiveVal()
    table_files <- reactiveVal()
    

    graphs_matrix <- reactive({
        
        #Initialize empty vectors to store the metadata, indexing represents each graph 
        num_classes_vect <- integer(input$num_graphs)
        num_epochs_vect <- integer(input$num_graphs)
        training_size_vect <- integer(input$num_graphs)
        model_arch_vect <- character(input$num_graphs)
        dataset_type_vect <- character(input$num_graphs)
        version_vect <- integer(input$num_graphs)
        #This list holds the vectors from the UI. Each index is a vector of the UI components for each graph line
        loss_checkbox_matrix <- list()
        #Initialize empty list of lists, each element holding a list of data.csv files, for each graph line
        graphs_data_vect <- list()
        #Loop through the number of graphs, and store the user-chosen metadata into the vectors
        for(i in 1:input$num_graphs){
            if(!is.null(input[[paste0("num_classes", i)]]) & !is.null(input[[paste0("num_epochs", i)]]) & 
            !is.null(input[[paste0("data_chunk", i)]]) & !is.null(input[[paste0("model_arch", i)]])
             & !is.null(input[[paste0("dataset_type", i)]]) & !is.null(input[[paste0("version", i)]])){ 

                print(input[[paste0("version", i)]])
                 #Vectors of metadata, each index corresponds to the # of line on graph 
                num_classes_vect[i]<- input[[paste0("num_classes", i)]]
                num_epochs_vect[i]<- input[[paste0("num_epochs", i)]]
                training_size_vect[i] <- input[[paste0("data_chunk", i)]]
                model_arch_vect[i] <- input[[paste0("model_arch", i)]]
                dataset_type_vect[i] <- input[[paste0("dataset_type", i)]]
                version_vect[i] <- input[[paste0("version", i)]]
                #If the metric is loss, and none of the input loss type values are null, store the checkbox vector
                if(input$metric == "loss" & !any(is.null(input[[paste0("loss_type", i)]]))){
            
                    loss_checkbox_matrix[[i]] <- input[[paste0("loss_type", i)]]
                }
                #If the metric is loss, but some values of the input are null, return error
                else if(input$metric == "loss" & any(is.null(input[[paste0("loss_type", i)]]))){
                    plot_viable(FALSE)
                    err_msg("Error: Invalid inputs")
                    return()
                }
                # If the metric is not loss, store NA in the loss checkbox matrix
                else{     
                    loss_checkbox_matrix[[i]] <- NA
                }  
            }
            #If some of the input values are null, return error
            else{
                err_msg("Error: Invalid inputs")
                plot_viable(FALSE)
                return()
            }
        }  
        #store the global values for num_epochs and loss_checkbox_matrix so they are accessible in other functions
        num_epochs_vect_global(num_epochs_vect)
        loss_checkbox_matrix_global(loss_checkbox_matrix)

        #Check that the inputs are valid 
        if(length(num_classes_vect) == input$num_graphs &
            length(num_epochs_vect) == input$num_graphs &
            length(training_size_vect) == input$num_graphs &
            length(model_arch_vect) == input$num_graphs &
            length(dataset_type_vect == input$num_graphs) &
            length(version_vect == input$num_graphs) &
            input$num_graphs > 0 & !any(is.na(num_classes_vect)) & 
            !any(is.na(num_epochs_vect)) & !any(is.na(training_size_vect)) 
            & !any(is.na(model_arch_vect)) & !any(is.na(dataset_type_vect)) & 
            !any(is.na(version_vect))){
            #Loop through the number of graphs, call the function to return the named list containing the metadata
            for(i in 1:input$num_graphs){
                #Fill the list with the list of dataframes (which match) for each graph line 
                matching_named_list <- get_matching_data(valid_data, "data.csv" ,num_classes_vect[i], num_epochs_vect[i],
                training_size_vect[i], model_arch_vect[i], dataset_type_vect[i], version_vect[i])

                graphs_data_vect[[i]] <- matching_named_list
                #Append the name of a matching file to the table_files reactiveVal
                matching_names <- names(matching_named_list)
                latest_matching_name <- matching_names[length(matching_names)]
                current_files <- table_files()  # Retrieve current value
                current_files[i] <- latest_matching_name  # Modify the value
                table_files(current_files)  # Update the reactive valvalue
          
     


                #If list in the list of lists is empty, this means that matching metadata was not found for that grpah line
                #And the plot cannot be generated, thus no data error is displayed
                if(length(graphs_data_vect[[i]]) == 0 || is.null(graphs_data_vect[[i]])){
                      # Display an error message within the plot area
                    plot_viable(FALSE)
                    err_msg("Error: No Data Found")
                    return()
                }
            }
            #If all is sucessfull, return the graphs matrix 
            plot_viable(TRUE)
            return(graphs_data_vect)
        }
        #If the inputs are invalid, return error
        else{
              # Display an error message within the plot area
            plot_viable(FALSE)
            err_msg("Error: Invalid inputs")
            return()
        }
    })
    # Render the plot 
    output$plot <- renderPlot({
        #Compute numepochs vect and loss checkbox matrix 
        #This line below ensures, renderPlot always runs when graphs_matrix() changes. This is crucial for reactivity
        graphs_data <- graphs_matrix()
        #If the plot is viable, generate either the single or ribbon plot 
        if(plot_viable()){
            main_plot(graphs_data, input$data_amount, input$metric, input$num_graphs, num_epochs_vect_global(), loss_checkbox_matrix_global())
        }
        # If not, throw an error message
        else{
            plot.new()
            title(main = err_msg(), col.main = "red", font.main = 4)
            return()
        }
    })

    output$dataframes <- renderUI({
         #This line below ensures, renderPlot always runs when table_files changes. This is crucial for reactivity
        jobs_list <- table_files()
        ui_components <- list()

        if(plot_viable()){
            for(i in 1:input$num_graphs){
                if(input$metric == "loss" & input$data_amount == "multi"){
                    for(j in seq_along(loss_checkbox_matrix_global()[[i]])){
                        loss_name <- name_change(loss_checkbox_matrix_global()[[i]][[j]])
                        print("this is the loss type")
                        print(loss_checkbox_matrix_global()[[i]][[j]])
                        ui_components <- c(ui_components, list( 
                            tags$h4(paste("Graph ", i, " ", loss_name)),
                            tableOutput(paste0("table", i, "_", loss_checkbox_matrix_global()[[i]][[j]]))
                        ))
                    }
                }
                else{
                    ui_components <- c(ui_components, list(
                        tags$h4(paste("Graph ", i)),
                        tableOutput(paste0("table", i))
                    )) 
                }
            }
             #Taglist replaces need for div
            do.call(tagList, ui_components)
        }
         else{
            print("Error: Invalid inputs")
            return()
        }
    })
   
    observe({
        jobs_list <- table_files()
        if(plot_viable()){
            if(input$data_amount == "single"){
                for(i in 1:input$num_graphs){
                    local({
                        local_i <- i

                        output[[paste0("table", local_i)]] <- renderTable({
                    
              
                        df <- search_directory_csv(paste0("../../job_outputs/", jobs_list[local_i]), 'data.csv')
                        colnames(df) <- c("Corruption Rate", "Epoch", "Accuracy", "Training Loss", "Test Loss", "Sensitivity", "Specificity")
                        return(df)
                        }) 
                    })
                
                } 
            }
            else{
                for(i in 1:input$num_graphs){
                    local({
                        local_i <- i
                        if(input$metric == "loss"){
                            for(j in seq_along(loss_checkbox_matrix_global()[[local_i]])){
                                local({
                                    local_j <- j 
                                    loss_type <- loss_checkbox_matrix_global()[[local_i]][[local_j]]
                                    print(jobs_list[local_i])
                                    output[[paste0("table", local_i,"_", loss_type)]] <- renderTable({
                                        summary_stats_df <- search_directory_csv(paste0("../../job_outputs/", jobs_list[local_i]), paste0("summary_stats_", loss_type, ".csv"))
                                        colnames(summary_stats_df)[1] <- "Corruption Rate"
                                        num_cols <- ncol(summary_stats_df)
                                        last_10_cols <- (num_cols - 9):num_cols
                                        colnames(summary_stats_df)[last_10_cols] <- c("Mean", "Median", "Range", "Standard Deviation", "Variance", "Minimum", "Maximum", "1st Quartile", "3rd Quartile", "Interquartile Range")
                                        return(summary_stats_df)
                                    })
                                })

                            }
                        }
                        else{
                            output[[paste0("table", local_i)]] <- renderTable({
                                summary_stats_df <- search_directory_csv(paste0("../../job_outputs/", jobs_list[local_i]), paste0("summary_stats_", input$metric, ".csv"))
                                colnames(summary_stats_df)[1] <- "Corruption Rate"
                                num_cols <- ncol(summary_stats_df)
                                last_10_cols <- (num_cols - 9):num_cols
                                colnames(summary_stats_df)[last_10_cols] <- c("Mean", "Median", "Range", "Standard Deviation", "Variance", "Minimum", "Maximum", "1st Quartile", "3rd Quartile", "Interquartile Range")
                                return(summary_stats_df)
                            })
                        }
                    })
                }
            }
        }
        else{
            return (data.frame(Error = "Error: unable to fetch data"))
        }
        
    })


    observe({
        num_buttons <- input$num_graphs
        for(i in 1:num_buttons){
            local({
                local_i <- i
                observeEvent(input[[paste0("job_run", local_i)]], {
                    num_epochs <- input[[paste0("num_epochs", local_i)]]
                    num_classes <- input[[paste0("num_classes", local_i)]]
                    data_chunk <- input[[paste0("data_chunk", local_i)]]
                    model_arch <- input[[paste0("model_arch", local_i)]]
                    dataset_type <- input[[paste0("dataset_type", local_i)]]
                    version <- input[[paste0("version", local_i)]]

                    bash_cmd <- paste("sbatch", bash_path, shQuote(num_epochs), shQuote(num_classes), 
                    shQuote(data_chunk), shQuote(dataset_type), shQuote(model_arch), shQuote(version))

                    isRejected <- system(bash_cmd, intern = FALSE)

                    if(!isRejected){
                        showModal(modalDialog(
                            title = "Job Submitted Successfully",
                            "Use the 'sq' command to monitor the status of the job",
                            easyClose = TRUE,
                            footer = modalButton("Close")
                        ))
                    }
                    else{
                        showModal(modalDialog(
                            title = "Job Failed to Submit",
                            "Check the parameters and contact your system admin to troubleshoot",
                            easyClose = TRUE,
                            footer = modalButton("Close")
                        ))
                    }

                    
                })
            })
        }
    })

    

    output$jobDetails <- renderUI({
        ui_components <- list()
        jobs_list <- table_files()

        
        for(i in 1:input$num_graphs){
            
            job_details <- getJobDetails(jobs_list[i])

            # Generate UI components for hyperparams
            hyperparamsUI <- recursiveFormatHyperparams(job_details$hyperparams)

            component <- list(
                tags$h4("Graph",i),
                tags$h4("Job ID", jobs_list[i]),
                tags$h5("Hyperparams:"),
                do.call(tagList,list(hyperparamsUI)),
                tags$h5("Criterion:", job_details$criterion),
                tags$h5("Scheduler:", job_details$scheduler),
                tags$h5("Optimizer:", job_details$optimizer),
                tags$br()
            )
            ui_components <- c(ui_components, component)
        }
         do.call(tagList, ui_components)
       
    })
    
}
# Run the app 
shinyApp(ui = ui, server = server)







