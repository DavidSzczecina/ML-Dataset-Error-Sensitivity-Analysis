

name_change <- function(metric){
    if(metric == "accuracy"){
        return("Accuracy (%)")
    }
    else if(metric == "avg_sens"){
        return("Average Sensitivity (%)")
    }
    else if(metric == "avg_spec"){
        return("Average Specificity (%)")
    }
    else if(metric == "test_loss"){
        return("Test Loss")
    }
    else if(metric == "training_loss"){
        return("Training Loss")
    }
    else if(metric == "loss"){
        return("Loss")
    }
    else{
        return("Error")
    }
}