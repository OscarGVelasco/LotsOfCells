#' Compute entropy score tests on single-cell data metadata.
#'
#' `entropyScore()` returns a dataframe containing the results and statistics for the given variables and covariables.
#'
#' This function will calculate a symmetric divergence score based on the Kullback- Leibler (KL) divergence. It will simulate a random distribution to try to understand how extreme is the observed symmetric score.
#'
#' @param scObject Object or DataFrame. An object of class Single Cell Experiments or Seurat, or a dataframe containing the metadata information.
#' @param main_variable Character. Name of the column on the metadata dataframe containing the main variable to be contrasted (e.g.: disease_status)
#' @param subtype_variable Character. Name of the column on the metadata dataframe containing the covariable to be analyzed (e.g.: cell_type, time_point, ...)
#' @param labelOrder Character Vector. Covariables found on the subtype_variable column in order to be contrasted (e.g.: c("type_A","type_B" will contrast type_A vs type_B))
#' @param sample_id Character. Column name containing the sample/patient id variable. If provided for tests, sampling will be done simulating the proportion variability per sample, for plots each individual will be shown.
#' @param permutations Numeric. Number of random permutations for the montecarlo test (2 label for main_variable case) or the gamma null distribution (main_variable > 2 labels) (default 1000).
#' @param parallel Boolean. Whether to use parallel computing with BiocParallel or not (default FALSE).
#'
#' @return The function returns a DataFrame containing the results and statistics tested for each covariables found in subtype_variable contrasting the main_variable labels. Column values are:
#' \tabular{ll}{
#'    \code{groupFC or groupGammaCor} \tab If groups = 2: FoldChange between groups in labelOrder order (labelOrder[1] / labelOrder[2]). If groups>2 the gamma rank correlation coeficient in the order determined by labelOrder \cr
#'    \tab \cr
#'    \code{percent_in_LABEL} \tab For each LABEL in main_variable, the percentage of each subtype_variable for each LABEL \cr
#'    \tab \cr
#'    \code{p.adj} \tab p.value adjusted using Bonferroni correction. \cr
#'    \tab \cr
#'    \code{sd.montecarlo} \tab If groups = 2: standard deviation of the fold changes simulated values. \cr
#'    \tab \cr
#'    \code{CI95low} \tab If groups = 2: lower 95\% confidence interval for the observed fold change (groupFC). \cr
#'    \tab \cr
#'    \code{CI95high} \tab If groups = 2: lower 95\% confidence interval for the observed fold change (groupFC). \cr
#' }
#'
#' @examples
#' # Montecarlo simulation of groups
#' # We define 2 main groups A and B with 5 different cell types with different cell numbers.
#' groups1 <- c(rep("CellA",700),rep("CellB",300),rep("CellC",500),rep("CellD",1000),rep("CellF",80))
#' groups2 <- c(rep("CellA",1700),rep("CellB",300),rep("CellC",550),rep("CellD",1200),rep("CellF",50))
#' groups <- c(rep("A",length(groups1)),rep("B",length(groups2)))
#' covariable <- c(groups1, groups2)
#' # We construct a metadata dataframe
#' meta.data <- data.frame(groups, covariable)
#' lotOfCell(scObject = meta.data,
#'           main_variable = "groups", # The column in meta.data to be used as the main variable (groups A and B)
#'           subtype_variable = "covariable", # The column in meta.data to be used as covariable (cell types)
#'           labelOrder = c("B","A"), # Order of the constrast, we will obtain the fold changes as: B vs A
#'           permutations = 1000)
#'
#' # Goodman and Kruskal's gamma rank correlation coeficient
#' # Data simulation with 3 or > groups
#' # In this example we define 4 different groups (A, B, C, D) with cell types
#' groups1 <- c(rep("CellA",700),rep("CellB",300),rep("CellC",500),rep("CellD",1000))
#' groups2 <- c(rep("CellA",1700),rep("CellB",350),rep("CellC",550),rep("CellD",800))
#' groups3 <- c(rep("CellA",1200),rep("CellB",200),rep("CellC",420),rep("CellD",800))
#' groups4 <- c(rep("CellA",500),rep("CellB",1000),rep("CellC",10),rep("CellD",1200))
#' groups <- c(rep("A",length(groups1)),rep("B",length(groups2)),rep("C",length(groups3)),rep("D",length(groups4)))
#' covariable <- c(groups1, groups2,groups3,groups4)
#' meta.data <- data.frame(groups, covariable)
#' lotOfCell(scObject = meta.data,
#'           main_variable = "groups",
#'           subtype_variable = "covariable",
#'           labelOrder = c("B","A","D","C"), # Order of the constrast, for gamma correlation it will test for an upward/downward pattern through B -> A -> D -> C
#'           permutations = 100)
#'
#' @author Oscar Gonzalez-Velasco
#' @importFrom stats sd p.adjust quantile
#' @importFrom SingleCellExperiment colLabels
#' @importFrom methods is
#' @importFrom BiocParallel bplapply
#' @importFrom ggbeeswarm geom_quasirandom
#' @export
entropyScore <- function(scObject=NULL, main_variable=NULL, subtype_variable=NULL, labelOrder=c(""), sample_id=NULL, permutations=1000, parallel=FALSE){
  # Obtain the single-cell metadata
  main_metadata <- getMetadata(scObject)
  # Test that all groups are in the data:
  if(isFALSE(all(labelOrder %in% unique(main_metadata[, main_variable])))){
    stop(paste("Some groups in labelOrder are not on the data:",paste(labelOrder, collapse = " ")))
  }
  # Subset only the main groups stated in labelOrder:
  main_metadata <- main_metadata[main_metadata[, main_variable] %in% labelOrder ,]
  groups <- as.character(main_metadata[, main_variable])
  covariable <- as.character(main_metadata[, subtype_variable])
  if(length(labelOrder)<2){
    stop("You have to specify the order of testing for the labels (labelOrder=c(label1,label2,labeln...)")
  }
  if(length(labelOrder)>2){
    stop(paste("Only 2 labels are allowed for computing the entropy estimation, found:",paste(labelOrder,collapse = ", ")))
  }
  functToApply <- base::lapply
  if(isTRUE(parallel)){
    functToApply <- BiocParallel::bplapply
  }
  # Calculate pseudocounts using the arcsin function
  pseudoCount <- function(counts){counts + sqrt((counts*counts)+1)}
  message(paste("Computing Entropy proportion over covariables for groups:",labelOrder[1],"vs",labelOrder[2]))
  df <- data.frame(groups, covariable)
  df <- table(df)
  contig_tab <- t(apply(pseudoCount(df), 1, function(row){row/(sum(row))}))[labelOrder, ]
  indexes <- colnames(contig_tab)
  relative_entropies <- apply(contig_tab,1,function(x){
    abs(log2((x[1]*log2(x[2])) / (x[1]*log2(x[1]))))
  })
  geom_mean <- function(x){exp(mean(log(x)))}
  # Test using the geometric mean to summarise the entropy distances scores across cell types
  distance_surprise <- function(p, q){
    return(geom_mean(abs(p*log2(p/q)))+geom_mean(abs(q*log2(q/p))) )
  }
  entropy_score <- distance_surprise(p=contig_tab[1,], q=contig_tab[2,])
  # Montecarlo test for random entropy distribution
  if(!is.null(sample_id)){
    samples <- as.character(main_metadata[, sample_id])
    nPerSample <- table(data.frame(groups, samples))[labelOrder,]
    nPerSample <- sqrt(nPerSample)
    cellCrowd <- apply(nPerSample, 1, function(perCond){list(perCond[perCond!=0])})
    cellCrowd <- cellCrowd[labelOrder]
  }else{
    cellCrowd <- round(sqrt(c(table(groups)))[labelOrder])
  }
  message(paste("Starting montecarlo simulation with n. permutations:",permutations))
  # for parallelization purposes it is better to split in 2 sub-loops so each thread has (permutation/10) computations
  entropy_list <- functToApply(seq(10),function(iteration){
    sapply(seq(permutations/10),function(permutation){
      if(is.list(cellCrowd)){
        cond1 <- lapply(cellCrowd[[1]][[1]], FUN = function(nCellToSample){sample(covariable ,size=nCellToSample)})
        cond1 <- unname(unlist(cond1))
        cond2 <- lapply(cellCrowd[[2]][[1]], FUN = function(nCellToSample){sample(covariable, size=nCellToSample)})
        cond2 <- unname(unlist(cond2))
        dftmp <- data.frame(covariable=c(cond1, cond2),
                            groups = c(rep(names(cellCrowd)[1],times=length(cond1)), rep(names(cellCrowd)[2],times=length(cond2))))

      }else{
        dftmp <- data.frame(covariable=c(sample(covariable,size = cellCrowd[1]),sample(covariable,size = cellCrowd[2])),
                            groups = c(rep(names(cellCrowd)[1],times=cellCrowd[1]),rep(names(cellCrowd)[2],times=cellCrowd[2])))
      }
      dftmp <- table(dftmp)
      if(!all(indexes %in% rownames(dftmp))){
        tmp <- do.call(rbind, rep(list(rep(0,ncol(dftmp))), sum(!(indexes %in% rownames(dftmp)))))
        rownames(tmp) <- indexes[!(indexes %in% rownames(dftmp))]
        dftmp <- rbind(dftmp, tmp)[indexes,]
      }
      # dftmp[dftmp == 0] = 1 * instead of using pseudocount 1 use arcsin:
      # Obtain Frequencies of classes
      contig_tab_random <- t(apply(pseudoCount(dftmp),2,function(row){row/(sum(row))}))[labelOrder, indexes]
      entropy_score <- distance_surprise(contig_tab_random[1,], contig_tab_random[2,])
    })
  })
  # Unpack results
  null_test_entropy <- unlist(entropy_list)
  sd_entropies <- sd(null_test_entropy)
  mean_entropies <- mean(null_test_entropy)
  median_entropies <- median(null_test_entropy)
  g.entropies <- ggplot2::ggplot(reshape2::melt(null_test_entropy), ggplot2::aes(y=value, x=factor(1))) +
    ggbeeswarm::geom_quasirandom(dodge.width = 0.3, varwidth = TRUE, stroke=NA, shape=16,
                                 colour = "#D5BADB",
                                 #fill= "#D5BADB"
                                 alpha = 0.5, size = 1.5) +
    ggplot2::geom_segment(ggplot2::aes(x = 0.85, y = median_entropies, xend = 1.15, yend = median_entropies),
                          color="#86608E",linewidth=0.5, alpha=0.3, inherit.aes=FALSE) +
    ggplot2::geom_point(ggplot2::aes(y=entropy_score, x=factor(1)), size=4, color="#F08080") +
    ggplot2::theme_classic() +
    ggplot2::theme(axis.ticks.x = ggplot2::element_blank(), axis.text.x = ggplot2::element_blank()) +
    ggplot2::xlab("") +
    ggplot2::ylab("symmetric divergence")
  # test 1 #
  # Calculate how extreme our observed values are in comparison with the random distribution
  # We want to see if the FoldChanges on the random distribution are lower or higher than the observed FoldChange
  p.vals <- sum(null_test_entropy >= entropy_score) / (permutations)
  #p.adj <- round(p.adjust(p = p.vals, method = "fdr"), digits = 6)
  g <- ggplot2::ggplot(reshape2::melt(contig_tab),ggplot2::aes(x=covariable,y=value,fill=factor(groups))) +
    ggplot2::geom_bar(stat="identity", position=ggplot2::position_dodge()) +
    ggplot2::scale_fill_brewer(palette="Blues") +
    ggplot2::theme_minimal() +
    ggplot2::guides(fill = guide_legend(title=paste("Class:", subtype_variable), drop=FALSE)) +
    ggplot2::ggtitle(paste("Symmetric Divergence Score:",round(entropy_score, digits = 3),"p.val.adj:",round(p.adj,digits = 3))) +
    ggplot2::theme(
      title = ggplot2::element_text(face="bold", size=ggplot2::rel(1.2)),
      strip.text=ggplot2::element_text(face="bold", size=ggplot2::rel(0.8)),
      axis.title.x=ggplot2::element_text(),
      axis.text.x=ggplot2::element_text(angle=45, vjust=1, hjust=1,size = 12)
    ) +
    ggplot2::ylab("proportion")
  gridExtra::grid.arrange(g,g.entropies,nrow=1,ncol=2,widths=c(1,0.3))
  return(c(relative_entropies,"entropy_score"=entropy_score,"p.val.adj"=p.adj, "mean.random.entropy"= mean_entropies, "sd.random.entropy"=sd_entropies))
}

# groups1 <- c(rep("CellA",700),rep("CellB",300),rep("CellC",500),rep("CellD",1000))
# groups2 <- c(rep("CellA",1700),rep("CellB",350),rep("CellC",550),rep("CellD",800))
# groups3 <- c(rep("CellA",1200),rep("CellB",200),rep("CellC",420),rep("CellD",800))
# groups4 <- c(rep("CellA",500),rep("CellB",1000),rep("CellC",10),rep("CellD",1200))
# groups <- c(rep("A",length(groups1)),rep("B",length(groups2)),rep("C",length(groups3)),rep("D",length(groups4)))
# labelOrder <- c("C","B","A","D")
# labelOrder <- c("D","B")
# covariable <- c(groups1, groups2,groups3,groups4)
# meta.data <- data.frame(groups, covariable)
# rownames(meta.data) <- as.character(1:nrow(meta.data))
#
# res1 <- entropyScore(scObject = meta.data,
#           main_variable = "groups",
#           subtype_variable = "covariable",
#           permutations = 10000,
#           labelOrder = labelOrder,
#           parallel = TRUE)
#
# res1[1:4]/log2(4)
#
# entropyScore(scObject = tmp,
#              main_variable = "status",
#              subtype_variable = "cell.type",
#              permutations = 10000,
#              labelOrder = c("KIF5A","Control"),
#              parallel = TRUE)
#
# entropyScore(scObject = tmp,
#              main_variable = "batch",
#              subtype_variable = "cell.type",
#              permutations = 10000,
#              labelOrder = c("1","2"),
#              parallel = TRUE)
