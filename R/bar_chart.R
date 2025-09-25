#' Create a barplot from single-cell metadata.
#'
#' `bar_chart()` returns a ggplot barplot to visualize percentages of a set of classes defined by the user.
#'
#' This function produces a barplot ggplot object with the data defined by the user.
#'
#' @param scObject Object or DataFrame. An object of class Single Cell Experiments or Seurat, or a dataframe containing the metadata information.
#' @param main_variable Character. Name of the column on the metadata dataframe containing the main variable to be used for splitting the dataset into the different bar groups (e.g.: disease_status)
#' @param subtype_variable Character. Name of the column on the metadata dataframe containing the covariable of interest that we want to visualize as percentages (e.g.: cell_type, time_point, ...)
#' @param sample_id Character. Column name containing the sample/patient id variable. If provided for tests, sampling will be done simulating the proportion variability per sample, for plots each individual will be shown.
#' @param subtype_only Character. Visualize only a specific class from subtype_variable. Useful if for example you only want to show the proportions of a specific cell type or subclass.
#' @param contribution Boolean. If a sample_id variable has been defined, whether to plot per sample contribution to the bar class with different shades of color (TRUE) or to split by sample_id in separated bars (default: FALSE)
#' @param colors Character vector. Vector of colors defined by the user to be used as palette. If more colors than specified are required, colorRampPalette will be used to create additional colors. If not specified the LotOfCells default color palette is used.
#'
#' @return The function returns a ggplot object with the barplot representing the population frequencies on the requested variables.
#'
#' @examples
#' # We construct a metadata dataframe
#' meta.data <- data.frame(condition, cell_type, sample)
#' bar_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type")
#' bar_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", sample_id = "sample")
#' bar_chart(meta.data, main_variable = "condition", subtype_variable = "cell_type", subtype_only = "CellType_E")
#' @author Oscar Gonzalez-Velasco
#' @importFrom stats sd p.adjust quantile
#' @importFrom SingleCellExperiment colLabels
#' @importFrom methods is
#' @importFrom ggpubr as_ggplot
#' @import ggplot2
#' @import dplyr
#' @import gridExtra
#' @export
bar_chart <- function(scObject = NULL,
                      main_variable = NULL,
                      subtype_variable = NULL,
                      sample_id = NULL,
                      subtype_only = NULL,
                      contribution = FALSE,
                      colors = NULL){
  library(dplyr)
  library(ggplot2)
  library(gridExtra)
  # Obtain the single-cell metadata
  main_metadata <- getMetadata(scObject)
  # #
  groups <- main_metadata[, main_variable]
  groups <- factor(groups, levels=sort(unique(groups)))
  covariable <- as.character(main_metadata[, subtype_variable])
  order <- names(sort(table(covariable),decreasing = FALSE))
  colorOrder <- rev(seq(1, length(order)))
  names(colorOrder) <- order

  # Define common Y scale breaks, minor breaks and labels.
  breaks_use <- seq(from = 0, to = 1, by = 0.1)
  minor_breaks_use <- seq(from = 0.05, to = 0.95, by = 0.1)
  labels_use <- seq(from = 0, to = 100, by = 10)

  if (!is.null(sample_id)){
    # Independent sample/factor has been specified:
    samples <- as.character(main_metadata[, sample_id])
    n.of.stack.bars <- table(unique(data.frame(groups, samples))$groups)
    names(n.of.stack.bars) <- names(table(unique(data.frame(groups, samples))$groups))
    groupSorting <- data.frame(groups, samples)
    groupSorting <- unique(groupSorting)
    levelsOrdered <- apply(groupSorting[order(groupSorting$groups),], 1, function(sortedElements){paste(sortedElements[1], sortedElements[2], sep="_")})
    ### IN PROGRESS - CONTRIBUTION
    if (contribution){
      # Contribution plot has been specified:

      # Generate color palette.
      colores <- getPalette(usePalette = colors,
                            nColors = length(colorOrder))
      colores <- rev(colores)
      names(colores) <- names(colorOrder)

      # Generate empty plot.
      empty_plot <- ggplot2::ggplot(data = data.frame(x = 1,
                                                      y = factor(names(colores),
                                                                 levels = names(colores))),
                                    mapping = ggplot2::aes(x = x,
                                                           y = y,
                                                           fill = y)) +
                    ggplot2::geom_bar(position = "stack",
                                      stat = "identity") +
                    ggplot2::scale_fill_manual(values = colores) +
                    ggplot2::guides(fill = ggplot2::guide_legend(title = paste("Class:", subtype_variable),
                                                                 drop = FALSE))
      plot_table <- ggplot2::ggplot_gtable(ggplot2::ggplot_build(empty_plot))
      legend_plot <- which(sapply(plot_table$grobs, function(x){x$name}) == "guide-box")
      legend <- plot_table$grobs[[legend_plot]]

      tmp <- data.frame(original = groups,
                        groups = paste(groups, samples, sep = "_"), covariable) %>%
             dplyr::group_split(original)

      tmp <- lapply(tmp, function(main_group){df <- data.frame(original = unique(main_group$original),
                                                               reshape2::melt(table(main_group$groups, main_group$covariable)))
                                              df$value <- df$value / sum(df$value)
                                              df$colores <- NA
                                              color_range <- log2(length(unique(df$Var1))) / 8 # Darkening and lightening factor depending on n of samples
                                              for (aColor in colores){aName <- names(colores)[colores %in% aColor]
                                                                      colors_use <- c(colorspace::lighten(aColor, color_range),
                                                                                      colorspace::darken(aColor, color_range))
                                                                      df$colores[df$Var2 %in% aName] <- c(grDevices::colorRampPalette(bias = 0.5,
                                                                                                                                      colors = colors_use)(length(unique(df$Var1))))}
                                              return(df)})

      contig_tab_resh <- do.call(rbind, tmp)
      colnames(contig_tab_resh) <- c("groups", "per_group", "covariable", "value", "colores")

      # Plot Bars
      contig_tab_resh[,"covariable"] <- factor(contig_tab_resh[,"covariable"], levels = order)
      contig_tab_resh$supra <- paste(contig_tab_resh$per_group, contig_tab_resh$covariable, sep = "_")
      colores <- contig_tab_resh$colores
      names(colores) <- contig_tab_resh$supra
      new_order <- unlist(lapply(names(colorOrder), function(type){contig_tab_resh$supra[contig_tab_resh$covariable %in% type]}))
      contig_tab_resh$supra <- factor(contig_tab_resh$supra, levels = new_order)

      # Generate the plot.
      g <- ggplot2::ggplot(data = contig_tab_resh,
                           mapping = ggplot2::aes(x = groups,
                                                  y = value,
                                                  group_by = supra,
                                                  fill = supra)) +
           ggplot2::geom_bar(position = "stack",
                             stat = "identity",
                             linewidth = NA) +
           ggplot2::theme_minimal() +
           ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45,
                                                              vjust = 1,
                                                              hjust = 1,
                                                              size = 12)) +
           ggplot2::scale_fill_manual(name = "supra",
                                      values = colores,
                                      drop = FALSE) +
           ggplot2::labs(title = paste("Proportions of", subtype_variable, "by", main_variable),
                         subtitle = paste("Contribution by", sample_id)) +
           ggplot2::scale_y_continuous(name = "percentage",
                                       breaks =  breaks_use,
                                       minor_breaks = minor_breaks_use,
                                       labels = labels_use) +
           ggplot2::theme(legend.position = "none")

      g <- gridExtra::arrangeGrob(g, legend, nrow = 1, widths = c(0.7,0.2))
      return(ggpubr::as_ggplot(g))
      ### IN PROGRESS - CONTRIBUTION
    } else {
      # No contribution plotting is specified:
      df <- data.frame(groups = paste(groups, samples, sep = "_"), covariable)
      contig_tab <- apply(table(df), 1, function(row){row/sum(row)})
      group_names <- colnames(contig_tab)
      colores <- getPalette(usePalette = colors,
                            nColors = length(colorOrder))
    }
  } else {
    # No independent sample/factor specified:
    df <- data.frame(groups, covariable)
    levelsOrdered <- levels(groups)
    n.of.stack.bars <- table(unique(data.frame(groups))$groups)
    names(n.of.stack.bars) <- names(table(unique(data.frame(groups))$groups))
    contig_tab <- apply(table(df),1,function(row){row/sum(row)})
    group_names <- colnames(contig_tab)
    colores <- getPalette(usePalette=colors, nColors = length(colorOrder))
  }

  # Plot Bars
  contig_tab_resh <- reshape2::melt(contig_tab)
  contig_tab_resh[,"covariable"] <- factor(contig_tab_resh[,"covariable"], levels = order)
  contig_tab_resh[,"groups"] <- factor(contig_tab_resh[,"groups"], levels = levelsOrdered)
  xmin.annotation <- c(0.5, cumsum(n.of.stack.bars)[1:length(n.of.stack.bars)-1]+0.5)
  xmax.annotation <- c(cumsum(n.of.stack.bars)[1:length(n.of.stack.bars)])+0.5
  annot.data <- data.frame(x=xmin.annotation, y=xmax.annotation, group=factor(names(n.of.stack.bars)))

  if (length(n.of.stack.bars) > 8){
    group_colores <- grDevices::colorRampPalette(colors = ggplot2::alpha(colour = RColorBrewer::brewer.pal(8, "Set2"), alpha = 0.8))(length(n.of.stack.bars))
    group_colores = colorspace::desaturate(col = group_colores, amount = 0.16)
  } else {
    group_colores <- suppressWarnings(ggplot2::alpha(colour = RColorBrewer::brewer.pal(length(n.of.stack.bars), "Set2"), alpha = 0.8))[1:length(n.of.stack.bars)]
    group_colores = colorspace::desaturate(col = group_colores, amount = 0.16)
  }

  if(!is.null(subtype_only)){
    groupForColors <- table(unlist(lapply(strsplit(x = group_names, split = "_"), function(group)group[1])))
    contig_tab_resh <- contig_tab_resh[contig_tab_resh[,"covariable"] %in% subtype_only, ]
    coloresSubtype <- colorspace::lighten(group_colores, amount = 0.4)
    coloresSubtype <- colorspace::desaturate(col = coloresSubtype, amount = 0.16)
    coloresSubtype <- coloresSubtype[1:length(groupForColors)]
    coloresSubtype <- rev(coloresSubtype)
    colores <- rep(coloresSubtype, groupForColors)
  }

  # Plot settings:
  if (!is.null(subtype_only)){
    annotate_ymin <- -0.02
    annotate_ymax <- -0.08
    annotate_y <- -0.05
    plot_subtitle <- paste("Class:", subtype_only)
    scale_y_continuous_limits <- c(-0.1, 1)
    legend.position <- "none"
    colors_use <- colores
  } else {
    annotate_ymin <- -0.02
    annotate_ymax <- -0.06
    annotate_y <- -0.04
    plot_subtitle <- ifelse(!is.null(sample_id), paste("Individual Sub-level by:", sample_id), "")
    scale_y_continuous_limits <- NULL
    legend.position <- NULL
    colors_use <- colores[colorOrder]
  }

  # Add Faceting. For future easy implementations and as a means to reorder bars within groups.
  contig_tab_resh$facets <- vapply(contig_tab_resh$groups, function(x){group_use <- strsplit(as.character(x), "_")[[1]][1]},
                                   FUN.VALUE = character(1))

  # Add Order to bars based on group with highest proportion.
  ## This assumes that the data table already encodes the levels in order and that the last level is the one with highest fraction.

  if (!is.null(sample_id)){
    # Order the bars in descending order based on the currently established groups (and order depicted by annot.data).
    levels_use <- contig_tab_resh %>%
                  dplyr::filter(covariable == rev(levels(contig_tab_resh$covariable))[1]) %>%
                  dplyr::mutate("facets" = factor(facets, levels = levels(annot.data$group))) %>%
                  dplyr::group_by(.data$facets) %>%
                  dplyr::arrange(dplyr::desc(value), .by_group = TRUE) %>%
                  dplyr::mutate(groups = as.character(groups)) %>%
                  dplyr::pull(groups)

  } else {
    levels_use <- contig_tab_resh %>%
                  dplyr::filter(covariable == rev(levels(contig_tab_resh$covariable))[1]) %>%
                  dplyr::mutate("facets" = factor(facets, levels = levels(annot.data$group))) %>%
                  dplyr::arrange(dplyr::desc(value), .by_group = TRUE) %>%
                  dplyr::mutate(groups = as.character(groups)) %>%
                  dplyr::pull(groups)

    # Recompute the annotation data with the new levels.
    annot.data$group <- levels_use
    annot.data$group <- factor(as.character(annot.data$group), levels = levels_use)
  }

  contig_tab_resh$groups <- factor(as.character(contig_tab_resh$groups), levels = levels_use)

  group_spacing <- split(levels(contig_tab_resh$groups), rep(1:length(n.of.stack.bars), n.of.stack.bars))
  group_spacing <- unlist(lapply(group_spacing, function(a)c(a,"skip")))
  group_spacing <- group_spacing[-1*length(group_spacing)]
  contig_tab_resh$pos <- 1:length(levels(contig_tab_resh$groups))
  contig_tab_resh$groups
  g <- ggplot2::ggplot(data = contig_tab_resh,
                       mapping = ggplot2::aes(x = groups,
                                              y = value,
                                              group_by = covariable,
                                              fill = if (!is.null(subtype_only)){groups} else {covariable})) +
       ggplot2::geom_bar(position = "stack",
                         stat = "identity") +
       ggplot2::theme_minimal() +
       ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45,
                                                          vjust = 1,
                                                          hjust = 1,
                                                          size = 12),
                      legend.position = legend.position,
                      strip.background = ggplot2::element_rect(color = "black", fill = "grey95"),
                      strip.text = ggplot2::element_text(color = "black", face = "bold")) +
       ggplot2::guides(fill = ggplot2::guide_legend(title = paste("Class:", subtype_variable),
                                                    drop = FALSE)) +
       ggplot2::labs(title = paste("Proportions of", subtype_variable, "by", main_variable),
                     subtitle = plot_subtitle) +
       ggplot2::scale_fill_manual(name = "covariable",
                                  values = colors_use,
                                  drop = FALSE) +
       ggplot2::annotate(ymin = annotate_ymin,
                         ymax = annotate_ymax,
                         xmin = xmin.annotation,
                         xmax = xmax.annotation,
                         geom = "rect",
                         fill = rev(group_colores)) +
       ggplot2::annotate(y = annotate_y,
                         x = rowMeans(annot.data[,c("x","y")]),
                         geom = "text",
                         label = annot.data$group,
                         color = "white",
                         fontface = "italic",
                         size = 4) +
       ggplot2::scale_y_continuous(name = "percentage",
                                   breaks = breaks_use,
                                   minor_breaks = minor_breaks_use,
                                   labels = labels_use,
                                   limits = scale_y_continuous_limits) #+
      # ggplot2::scale_x_discrete(breaks = levels(contig_tab_resh$groups),
      #                           limits = group_spacing)
      # ADD SPACE BETWEEN GROUPS - IN progress
  return(g)
}
