#' Create a density plot from single-cell metadata.
#'
#' `density_chart()` returns a ggplot multi-density plot to visualize numerical values over a set of classes defined by the user.
#'
#' This function produces a multi-density ggplot object with the data defined by the user.
#'
#' @param scObject Object or DataFrame. An object of class Single Cell Experiments or Seurat, or a dataframe containing the metadata information.
#' @param main_variable Character. Name of the column on the metadata dataframe containing the main variable to be used for splitting the dataset into the different density groups (e.g.: disease_status)
#' @param subtype_variable Character. Name of the column on the metadata dataframe containing the covariable that we want to split as second level density group (e.g.: cell_type, time_point, ...)
#' @param numerical_variable Character. Name of the column on the metadata dataframe containing the numerical variable of interest that we want to plot the density distribution (e.g.: N_features_expressed)
#' @param sample_id Character. Column name containing the sample/patient id variable. If provided for tests, sampling will be done simulating the proportion variability per sample, for plots each individual will be shown.
#' @param subtype_only Character. Visualize only a specific class from subtype_variable. Useful if for example you only want to show the proportions of a specific cell type or subclass.
#' @param colors Character vector. Vector of colors defined by the user to be used as palette. If more colors than specified are required, colorRampPalette will be used to create additional colors. If not specified the lotsOfCells default color palette is used.
#'
#' @return The function returns a ggplot object with the barplot representing the population frequencies on the requested variables.
#'
#' @examples
#' # We construct a metadata dataframe
#' meta.data <- data.frame(condition, cell_type, sample)
#' density_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", numerical_variable = "nFeature_originalexp")
#' density_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", numerical_variable = "nFeature_originalexp",
#'                         sample_id = "sample")
#' density_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", numerical_variable = "nFeature_originalexp",
#'                         subtype_only = "CellType_E")
#' @author Oscar Gonzalez-Velasco
#' @importFrom stats sd p.adjust quantile
#' @importFrom SingleCellExperiment colLabels
#' @importFrom Seurat DefaultAssay
#' @importFrom methods is
#' @importFrom ggpubr as_ggplot
#' @import ggplot2
#' @import dplyr
#' @import ggridges
#' @export
density_chart <- function(scObject=NULL, main_variable=NULL, subtype_variable=NULL, numerical_variable=NULL, sample_id=NULL, subtype_only=NULL, colors=NULL){
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  # Obtain the single-cell metadata
  main_metadata <- getMetadata(scObject)
  # #
  # Remove NAs:
  main_metadata <- main_metadata[!is.na(main_metadata[, subtype_variable]),]
  groups <- main_metadata[, main_variable]
  groups <- factor(groups, levels=sort(unique(groups)))
  covariable <- as.character(main_metadata[, subtype_variable])
  order <- names(sort(table(covariable),decreasing = FALSE))
  main_metadata[, subtype_variable] <- factor(main_metadata[, subtype_variable], levels = order)
  # colorOrder <- rev(seq(1, length(order))) # Original order
  colorOrder <- seq(1, length(order))
  names(colorOrder) <- order
  numerical_variable <- numerical_variable[1]
  if(!numerical_variable %in% colnames(main_metadata)){
    # search in the scObject in case it is a feature:
    if(numerical_variable %in% rownames(scObject)){
     # Found on features:
      if(is(scObject, 'Seurat')){
      main_metadata[,numerical_variable] <- scObject[[Seurat::DefaultAssay(scObject)]]$counts[numerical_variable, rownames(main_metadata)]
      }
      if(is(scObject, 'SingleCellExperiment')){
      main_metadata[,numerical_variable] <- SingleCellExperiment::counts(scObject)[numerical_variable, rownames(main_metadata)]
      }
    } else stop("Defined variable (", numerical_variable, ") not found on scObject.")
    } else
    if(!is.numeric(main_metadata[, numerical_variable])){
    stop("Defined variable (", numerical_variable, ") is not numerical.")
      }
  main_metadata <- as.data.frame(main_metadata)
  # Remove NAs:
  main_metadata <- main_metadata[!is.na(main_metadata[, numerical_variable]),]
  if(is.null(sample_id)){
    # No sample level specified:
    plotOrder <- unlist(lapply(order, function(labelToPaste){paste(labelToPaste, levels(groups), sep = "_")}))
    main_metadata$plotLabel <- paste(covariable, groups, sep = "_")
    main_metadata$plotLabel <- factor(main_metadata$plotLabel, levels = plotOrder)

    colores <- getPalette(usePalette=colors, nColors=length(colorOrder))
    colores <- rev(colores)
    names(colores) <- names(colorOrder)
    # Set distinct shadows per group?
    main_metadata[,main_variable] <- factor(main_metadata[,main_variable])
    if(length(levels(groups)) < 4){alphaFactor <- 0.2} else
      if(length(levels(groups)) < 8){alphaFactor <- 0.1} else
      {alphaFactor <- 0.05}
    alphasToSet <- rev(c(0.9, 0.9 - cumsum(rep(alphaFactor, length(levels(groups))-1))))
    names(alphasToSet) <- levels(groups)
    main_metadata$setAlpha <- alphasToSet[main_metadata[,main_variable]]

    #names(colores) <- levels(main_metadata$plotLabel)
    group_spacing <- split(levels(main_metadata$plotLabel), ceiling(seq_along(levels(main_metadata$plotLabel))/length(levels(groups))))
    group_spacing <- unlist(lapply(group_spacing, function(a)c(a,"skip")))
    group_spacing <- group_spacing[-1*length(group_spacing)]
    g <- ggplot2::ggplot(main_metadata, ggplot2::aes(x = .data[[numerical_variable]], y = plotLabel,
                                                 fill = ggplot2::stat(.data[[subtype_variable]]),
                                                alpha=setAlpha)) +
      ggridges::geom_density_ridges(scale = 2, rel_min_height = 0.01,
                                             quantile_lines = TRUE, quantiles = 2) +
      ggplot2::scale_x_continuous(expand = c(0.01, 0)) +
      ggridges::theme_ridges(font_size = 13, grid = TRUE) +
      ggplot2::scale_fill_manual(values = colores, name = subtype_variable, breaks=rev(c(order))) +
      ggplot2::scale_alpha_identity() +
      ggplot2::labs(title = paste('Density distribution of', numerical_variable,"across", subtype_variable)) +
      ggplot2::theme(axis.title.y = ggplot2::element_blank()) +
      ggplot2::scale_y_discrete(breaks = levels(main_metadata$plotLabel),
                       limits = group_spacing)

  }else{
    # Sample level specified:
    samples <- as.character(main_metadata[, sample_id])
    groupSorting <- data.frame(groups, samples)
    groupSorting <- unique(groupSorting)
    levelsOrdered <- apply(groupSorting[order(groupSorting$groups),], 1, function(sortedElements)paste(sortedElements[1], sortedElements[2], sep="_"))

    plotOrder <- unlist(lapply(order, function(labelToPaste){paste(labelToPaste, levelsOrdered, sep = "_")}))
    main_metadata$plotLabel <- paste(covariable, groups, samples, sep = "_")

    main_metadata$plotLabel <- factor(main_metadata$plotLabel, levels = plotOrder)
    if(!is.numeric(main_metadata[, numerical_variable])){
      stop("Defined variable (", numerical_variable, ") is not numerical.")
    }

    colores <- getPalette(usePalette=colors, nColors=length(colorOrder))
    colores <- rev(colores)
    names(colores) <- names(colorOrder)
    # Set distinct shadows per group?
    main_metadata[,main_variable] <- factor(main_metadata[,main_variable])
    if(length(levels(groups)) < 4){alphaFactor <- 0.2} else
      if(length(levels(groups)) < 8){alphaFactor <- 0.1} else
        {alphaFactor <- 0.05}
    alphasToSet <- rev(c(0.9, 0.9 - cumsum(rep(alphaFactor, length(levels(groups))-1))))
    names(alphasToSet) <- levels(groups)
    main_metadata$setAlpha <- alphasToSet[main_metadata[,main_variable]]
    # Add small spacing in Y axis between groups:
    group_spacing <- split(levels(main_metadata$plotLabel), ceiling(seq_along(levels(main_metadata$plotLabel))/nrow(groupSorting)))
    group_spacing <- unlist(lapply(group_spacing, function(a)c(a,"skip")))
    group_spacing <- group_spacing[-1*length(group_spacing)]
    g <- ggplot2::ggplot(main_metadata, ggplot2::aes(x = .data[[numerical_variable]], y = plotLabel,
                                                     group_by=.data[[main_variable]], fill = ggplot2::stat(.data[[subtype_variable]]),
                                                     alpha=setAlpha)) +
      ggridges::geom_density_ridges(scale = 2, rel_min_height = 0.01,
                                             quantile_lines = TRUE, quantiles = 2) +
      ggplot2::scale_x_continuous(expand = c(0.01, 0)) +
      # ggplot2::scale_y_discrete(expand = c(0.01, 0)) +
      ggridges::theme_ridges(font_size = 13, grid = TRUE) +
      ggplot2::scale_fill_manual(values = colores, name = subtype_variable, breaks=rev(c(order))) +
      ggplot2::scale_alpha_identity() +
      ggplot2::labs(title = paste('Density distribution of', numerical_variable,"across", subtype_variable),
                    subtitle = paste("split across", sample_id)) +
      ggplot2::theme(axis.title.y = ggplot2::element_blank()) +
      ggplot2::scale_y_discrete(breaks = levels(main_metadata$plotLabel),
                                limits = group_spacing)

  }

  return(g)
}
