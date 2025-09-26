#' Create a polar plot from single-cell metadata.
#'
#' `polar_chart()` returns a ggplot polar plot to visualize percentages of a set of classes defined by the user.
#'
#' This function produces a polar ggplot object with the data defined by the user.
#'
#' @param scObject Object or DataFrame. An object of class Single Cell Experiments or Seurat, or a dataframe containing the metadata information.
#' @param main_variable Character. Name of the column on the metadata dataframe containing the main variable to be used for splitting the dataset into the different bar groups (e.g.: disease_status)
#' @param subtype_variable Character. Name of the column on the metadata dataframe containing the covariable of interest that we want to visualize as percentages (e.g.: cell_type, time_point, ...)
#' @param sample_id Character. Column name containing the sample/patient id variable. If provided for tests, sampling will be done simulating the proportion variability per sample, for plots each individual will be shown.
#' @param subtype_only Character. Visualize only a specific class from subtype_variable. Useful if for example you only want to show the proportions of a specific cell type or subclass.
#' @param colors Character vector. Vector of colors defined by the user to be used as palette. If more colors than specified are required, colorRampPalette will be used to create additional colors. If not specified the lotsOfCells default color palette is used.
#'
#' @return The function returns a ggplot object with the polar plot representing the population frequencies on the requested variables.
#'
#' @examples
#' # We construct a metadata dataframe
#' meta.data <- data.frame(condition, cell_type, sample)
#' polar_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type")
#' polar_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", sample_id = "sample")
#' polar_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", subtype_only = "CellType_E")
#' @author Oscar Gonzalez-Velasco
#' @importFrom stats sd p.adjust quantile
#' @importFrom SingleCellExperiment colLabels
#' @importFrom methods is
#' @import ggplot2
#' @import dplyr
#' @import gridExtra
#' @export
polar_chart <- function(scObject=NULL, main_variable=NULL, subtype_variable=NULL, sample_id=NULL, subtype_only=NULL, colors=NULL){
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  # Obtain the single-cell metadata
  main_metadata <- getMetadata(scObject)
  # #
  groups <- as.character(main_metadata[, main_variable])
  covariable <- as.character(main_metadata[, subtype_variable])
  order <- names(sort(table(covariable),decreasing = FALSE))
  colorOrder <- rev(seq(1, length(order)))
  names(colorOrder) <- order
  if(!is.null(sample_id)){
    samples <- as.character(main_metadata[, sample_id])
    df <- data.frame(groups,covariable,samples)
    df <- data.frame(groups=paste(groups,samples,sep = "_"), covariable)
    contig_tab <- t(table(df))
  }else{
    df <- data.frame(groups, covariable)
    contig_tab <- t(table(df))
  }
  group_names <- colnames(contig_tab)
  colores <- getPalette(usePalette=colors, nColors = length(colorOrder))
  # Plot the circos
  contig_tab_resh <- reshape2::melt(contig_tab)
  contig_tab_resh[,"covariable"] <- factor(contig_tab_resh[,"covariable"], levels = order)
  if(!is.null(subtype_only)){
    contig_tab_resh <- contig_tab_resh[contig_tab_resh[,"covariable"] %in% subtype_only, ]
  }
  contig_tab_resh$main_group <- factor(unlist(lapply(strsplit(as.character(contig_tab_resh$groups), "_"),function(element)element[[1]])))
  # Set a number of 'empty bar' to add at the end of each group
  empty_bar <- 2

  to_add <- data.frame( matrix(NA, empty_bar*nlevels(contig_tab_resh$main_group), ncol(contig_tab_resh)) )
  colnames(to_add) <- colnames(contig_tab_resh)
  to_add$main_group <- rep(levels(contig_tab_resh$main_group), each=empty_bar)
  to_add$groups = rep(paste("dummy",seq(empty_bar*nlevels(contig_tab_resh$main_group)/empty_bar),sep = "_"), each=empty_bar)
  contig_tab_resh <- rbind(contig_tab_resh, to_add)
  contig_tab_resh <- contig_tab_resh %>% arrange(main_group)
  contig_tab_resh$id <- factor(unlist(lapply(1:length(unique(contig_tab_resh$groups)),function(times)rep(times, table(contig_tab_resh$groups)[unique(contig_tab_resh$groups)][times]))))
  labels <- as.character(unique(contig_tab_resh$groups))
  labels[grep(pattern = "dummy_", labels)] <- ""
  p1 <- ggplot2::ggplot(contig_tab_resh, ggplot2::aes(x=id, y=value, group_by=covariable, fill = covariable)) +
    ggplot2::geom_bar(position="stack", stat="identity")
  originalY <- ggplot_build(p1)$layout$panel_params[[1]]$y$breaks
  originalY <- originalY[-1]

  g <- ggplot2::ggplot(contig_tab_resh, ggplot2::aes(x=id, y=value, group_by=covariable, fill = covariable)) +
    ggplot2::geom_bar(position="stack", stat="identity") +
    ggplot2::theme_minimal() +
    ggplot2::theme(
      legend.position = "none",
      axis.title.x = ggplot2::element_blank(),
      axis.title.y = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      axis.text.y = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(face = "bold"),
      plot.title = ggplot2::element_text(size = 24, face = "bold"),
      plot.subtitle = ggplot2::element_text(size = 12)
    ) +
    ggplot2::scale_fill_manual(values = colores[colorOrder], drop=FALSE) +
    ggplot2::ggtitle(paste("Proportions of", subtype_variable, "by", main_variable)) +
    ggplot2::ylab("") + ggplot2::xlab("") +
    ggplot2::coord_polar() +
    ggplot2::scale_x_discrete(labels=labels) +
    ggplot2::scale_y_continuous(limits = c(-min(colSums(contig_tab))/2, NA)) +
    ggplot2::annotate('text', x = 0, y = originalY, label = paste0("italic(", as.character(originalY),")"), size=3,
                      parse = TRUE)
  return(g)
}
