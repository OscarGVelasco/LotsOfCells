#' Create Waffle plots from single-cell metadata.
#'
#' `waffle_chart()` returns an arrange of ggplots of waffle plots to visualize percentages of a set of classes defined by the user.
#'
#' This function produces a series of waffle plots with the data defined by the user. Each square/tile represents a 1% of the population.
#'
#' @param scObject Object or DataFrame. An object of class Single Cell Experiments or Seurat, or a dataframe containing the metadata information.
#' @param main_variable Character. Name of the column on the metadata dataframe containing the main variable to be used for splitting the dataset into separated waffle plots (e.g.: disease_status)
#' @param subtype_variable Character. Name of the column on the metadata dataframe containing the covariable of interest that we want to visualize as percentages (e.g.: cell_type, time_point, ...)
#' @param sample_id Character. Column name containing the sample/patient id variable. If provided for tests, sampling will be done simulating the proportion variability per sample, for plots each individual will be shown.
#' @param subtype_only Character. Visualize only a specific class from subtype_variable. Useful if for example you only want to show the proportions of a specific cell type or subclass.
#' @param colors Character vector. Vector of colors defined by the user to be used as palette. If more colors than specified are required, colorRampPalette will be used to create additional colors. If not specified the lotsOfCells default color palette is used.
#'
#' @return The function returns a grid arrange of ggplots with the waffle plots representing the population frequencies on the requested variables.
#'
#' @examples
#' # We construct a metadata dataframe
#' meta.data <- data.frame(condition, cell_type, sample)
#' waffle_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type")
#' waffle_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", sample_id = "sample")
#' waffle_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", subtype_only = "CellType_E")
#' @author Oscar Gonzalez-Velasco
#' @importFrom stats sd p.adjust quantile
#' @importFrom SingleCellExperiment colLabels
#' @importFrom methods is
#' @import ggplot2
#' @import dplyr
#' @import gridExtra
#' @export
waffle_chart <- function(scObject=NULL, main_variable=NULL, subtype_variable=NULL, sample_id=NULL, subtype_only=NULL, colors=NULL){
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  # Obtain the single-cell metadata
  main_metadata <- getMetadata(scObject)
  groups <- as.character(main_metadata[, main_variable])
  covariable <- as.character(main_metadata[, subtype_variable])
  order <- rev(names(sort(table(covariable),decreasing = TRUE)))
  colorOrder <- rev(seq(1, length(order)))
  names(colorOrder) <- order
  if(!is.null(subtype_only)){
    if(!subtype_only %in% covariable){
      stop("The class specified in subtype_only does not exist in subtype_variable.")
    }
    subtype_only <- subtype_only[1]
    covariable[!covariable==subtype_only] <- "All Other"
    order <- c(subtype_only, "All Other")
    colorOrder <- c(2, 1)
    names(colorOrder) <- order
  }
  if(!is.null(sample_id)){
    samples <- as.character(main_metadata[, sample_id])
    nPerGroup <- cumsum(colSums(table(samples, groups)!=0))
    df <- data.frame(groups=paste(groups,samples,sep = "_"), covariable)
    if(!is.null(subtype_only)){
      ncells <- table(df)[,subtype_only]
    }else{
      ncells <- rowSums(table(df))
    }
    ncells <- format(ncells, big.mark = ",", scientific = F)
    contig_tab <- apply(table(df),1,function(row){row/sum(row)})
  }else{
    df <- data.frame(groups, covariable)
    nPerGroup <- table(groups)
    nPerGroup <- setNames(rep(1, length(nPerGroup)), names(nPerGroup))
    if(!is.null(subtype_only)){
      ncells <- table(df)[,subtype_only]
    }else{
      ncells <- rowSums(table(df))
    }
    ncells <- format(ncells, big.mark = ",", scientific = F)
    contig_tab <- apply(table(df), 1, function(row){row/sum(row)})
  }
  group_names <- colnames(contig_tab)
  if(!is.null(subtype_only)){
    coloresSubtype = scales::alpha(c("#DBECDA","#92C791","#BEDAEC","#7EB6D9","#DDC7E2","#86608E"), 0.8)
    if(length(nPerGroup)>3){
      group_colores_l <- colorspace::lighten(grDevices::colorRampPalette(colors = ggplot2::alpha(colour = RColorBrewer::brewer.pal(8, "Set2"),
                                                                              alpha = 0.8))(length(nPerGroup)-3), amount = 0.6)
      group_colores_d <- colorspace::lighten(grDevices::colorRampPalette(colors = ggplot2::alpha(colour = RColorBrewer::brewer.pal(8, "Set2"),
                                                                              alpha = 0.8))(length(nPerGroup)-3), amount = 0.2)
      coloresSubtype <- c(coloresSubtype, c(rbind(group_colores_l, group_colores_d)))
      coloresSubtype <- colorspace::desaturate(col = coloresSubtype, amount = 0.16)
    }
  }else{
    colores <- getPalette(usePalette=colors, nColors = length(colorOrder))
  }
  # Plot the waffles
  plotingGroupN <- 1
  g.list <- lapply(seq(ncol(contig_tab)), function(indx){
    percentages <- contig_tab[order, indx]*100
    colour <- names(percentages)
    if(!is.null(subtype_only)){
      plotingGroupN <- (plotingGroupN + sum(indx > cumsum(nPerGroup)))*2
      colores <- coloresSubtype[c(plotingGroupN-1, plotingGroupN)]
    }
    df.p <- expand.grid(x = 0:9,
                        y = 0:9) %>%
      dplyr::rowwise() |>
      dplyr::mutate(index=1 + sum(x * 10 + y >= cumsum(percentages)),
             col = colour[[index]]) %>%
      as.data.frame()

    df.p[,"col"] <- factor(df.p[,"col"], levels = order)
    gp <- ggplot(df.p, ggplot2::aes(x=y,y=x,fill = col)) +
      ggplot2::geom_tile(aes(width = 0.85, height = 0.85)) +
      ggplot2::coord_equal() +
      ggplot2::theme_void() +
      ggplot2::theme(plot.caption = element_text(color = "grey", face = "italic", vjust=4, size=12),
                     legend.margin=margin(c(1, 1, 1, 1)),
                     plot.margin = margin(t=-1, r=1, b=-1, l=0)) +
      ggplot2::scale_fill_manual(values = colores[colorOrder], drop=FALSE) +
      ggplot2::guides(fill = guide_legend(title=paste("Class:", subtype_variable), drop=FALSE)) +
      ggplot2::ggtitle(group_names[indx]) +
      ggplot2::labs(caption = paste("n. cells:", ncells[group_names[indx]]))
    if(!is.null(subtype_only)){
      gp <- gp + ggplot2::geom_label(aes(x = 5, y = 8, label = paste0(round(percentages[1], 2),"%")),
                                     fill = "white", label.size = NA, size = 4, color=colores[2], fontface = "bold")
    }
    return(gp)
  })
  # function to extract legend from plot
  get_only_legend <- function(plot) {
    plot_table <- ggplot2::ggplot_gtable(ggplot_build(plot))
    legend_plot <- which(sapply(plot_table$grobs, function(x) x$name) == "guide-box")
    legend <- plot_table$grobs[[legend_plot]]
    return(legend)
  }
  # extract legend from plot1 using above function
  if(!is.null(subtype_only)){
    tmpLegend <- ggplot2::ggplot(data = data.frame(Class=factor(names(colorOrder)), y=1:2), ggplot2::aes(x=Class,y=y,fill=Class)) +
      ggplot2::geom_tile() + ggplot2::scale_fill_manual(values = scales::alpha(c("#EEEEEE", "#5F5F5F"),alpha = 0.8), drop=FALSE)
    legend <- get_only_legend(tmpLegend)
  }else{
    legend <- get_only_legend(g.list[[1]])
  }
  g.list <- lapply(g.list, function(plot)plot+ggplot2::theme(legend.position = "none"))
  if(is.null(sample_id)){
    # If no plot per sample we assume a much smaller number of groups:
    multiplot <- do.call("arrangeGrob", c(g.list, nrow=round(sqrt(length(unique(groups))))))
    }
  else{
    # Minimum 2 rows
    multiplot <- do.call("arrangeGrob", c(g.list,nrow=max(round(sqrt(length(unique(groups)))),2)))
  }
  #  return(grid.arrange(multiplot, legend, ncol = 2, heights = c(10, 1.5), widths = c(10,1.5)))
  return(grid.arrange(multiplot, legend, ncol = 2, heights = c(0.9, 0.1), widths = c(0.9, 0.1)))
}

