recursiveFormatHyperparams <- function(hyperparam) {
  if (is.list(hyperparam)) {
    ui_elements <- lapply(names(hyperparam), function(name) {
      value <- hyperparam[[name]]
      if (is.list(value)) {
        # Nested list, apply recursion
        nested_ui <- recursiveFormatHyperparams(value)
        do.call(tagList, c((name), nested_ui))
      } else if (is.atomic(value)) {
        # Simple element, display name and value
        tags$p(paste(name, ":", toString(value)))
      } else {
        # Fallback for other types
        tags$p(paste(name, ": [Complex Type]"))
      }
    })
    do.call(tagList, ui_elements)
  } else {
    # For atomic types, return the value directly
    tags$p(toString(hyperparam))
  }
}

# ... rest of your Shiny app code remains the same
