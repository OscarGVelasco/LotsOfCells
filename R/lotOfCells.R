#' Compute proportion tests on single-cell data metadata.
#'
#' `lotOfCells()` returns a dataframe containing the results and statistics for the given variables and covariables.
#'
#' This function will calculate a fold change of the proportions if two classes are specified, if more than two ordered classes are specified a Kendall rank correlation coefficient will be computed (e.g.: c("type-A", type_B", type_C") will compute a rank correlation to test whether proportions increase or decrease concordantly from A to B and from B to C).
#' In both cases a Montecarlo simulation to compute coefficients from a random sampling of the whole population will be done, and a p.value of how extreme is the observed score compared with the simulation values will be computed. A plot will be produced with the results, in the case of two class comparison, a pink shade is drawn to show the Montecarlo simulation standard deviation.
#'
#' @param scObject Object or DataFrame. An object of class Single Cell Experiments or Seurat, or a dataframe containing the metadata information.
#' @param main_variable Character. Name of the column on the metadata dataframe containing the main variable to be contrasted (e.g.: disease_status)
#' @param subtype_variable Character. Name of the column on the metadata dataframe containing the covariable to be analyzed (e.g.: cell_type, time_point, ...)
#' @param labelOrder Character Vector. Covariables found on the subtype_variable column in the order to be contrasted (e.g.: c("type_A","type_B" will contrast type_A vs type_B))
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
#' lotOfCells(scObject = meta.data,
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
#' lotOfCells(scObject = meta.data,
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
#' @export
lotOfCells <- function(scObject=NULL, main_variable=NULL, subtype_variable=NULL, labelOrder=c(""), sample_id=NULL, permutations=1000, parallel=FALSE){
  # Obtain the single-cell metadata
  main_metadata <- getMetadata(scObject)
  # Test that all groups are in the data:
  if(isFALSE(all(labelOrder %in% unique(main_metadata[, main_variable])))){
    stop(paste("Some groups in labelOrder are not on the data:",paste(labelOrder, collapse = " ")))
  }
  functToApply <- base::lapply
  if(isTRUE(parallel)){
    functToApply <- BiocParallel::bplapply
    ncores <- BiocParallel::bpnworkers(BiocParallel::bpparam())
  }
  # Subset only the main groups stated in labelOrder:
  main_metadata <- main_metadata[main_metadata[, main_variable] %in% labelOrder ,]
  groups <- as.character(main_metadata[, main_variable])
  covariable <- as.character(main_metadata[, subtype_variable])
  pseudoCount <- function(counts){counts + sqrt(counts^2+1)}
  # ###
  # covariable <- milk.metadata$General_Celltype
  # groups <- as.character(as.numeric(milk.metadata$time_post_partum_days))
  # labelOrder <- as.character(sort(unique(as.numeric(milk.metadata$time_post_partum_days))))
  # ###
  if(length(labelOrder)<2){
    stop("You have to specify the order of testing for the labels (labelOrder=c(label1,label2,labeln...)")
  }
  # Check Number of groups in main covariable
  if(length(labelOrder)>2){
    ## Gamma Rank Correlation Analysis #
    # Set variables for the random permutation test:
    message("More than 2 groups detected.")
    message(paste("Computing Goodman and Kruskal's gamma rank correlation coefficient in the following order:",paste(labelOrder,collapse = ' vs ')))
    cellCrowd <- round(sqrt(c(table(groups))))
    cellCrowd <- cellCrowd[labelOrder]
    kendallDenominator <- (length(labelOrder) * (length(labelOrder)-1)) / 2 #n. of pairs to be compared: (N * (N-1)) / 2
    # Proportions
    df <- data.frame(groups, covariable)
    contig_tab <- t(apply(table(df),1,function(row){row/sum(row)}))[labelOrder,]
    indexes <- colnames(contig_tab)
    rank_index <- c(1:length(labelOrder))
    # Goodman and Kruskal's gamma function:
    godKrusGamma <- function(nc, nd, N){(nc-nd)/exp(mean(log(N)))}
    # Call gamma rank correlation to perform a permutation test on the original sets
    # PARALLEL FINE TUNING - CONSTRUCTION
    original_gamma_test <- functToApply(seq_len(round(sqrt(permutations))),function(x){
      original_gamma_test_sub <- lapply(seq_len(permutations/round(sqrt(permutations))),function(y){
        cellToGammaOriginal(covariable, groups, labelOrder, indexes, cellCrowd, rank_index)
        })
    })
    original_concordant <- colSums(do.call(rbind, lapply(original_gamma_test,function(l)do.call(rbind,lapply(l,function(results)results[[1]])))))
    original_disconcordant <- colSums(do.call(rbind, lapply(original_gamma_test,function(l)do.call(rbind,lapply(l,function(results)results[[2]])))))
    original_gamma_cor <- mapply(FUN = godKrusGamma, original_concordant, original_disconcordant,kendallDenominator*permutations)
    # Calculate the Confidence Interval for the gamma cor:
    subsampled_gamma_cor <- functToApply(seq_len(10), function(subsample){
      subsample_gamma_test <- lapply(seq_len(100), function(x){
        cellToGammaOriginal(covariable, groups, labelOrder, indexes, cellCrowd, rank_index)})
      subsample_concordant <- colSums(do.call(rbind, lapply(subsample_gamma_test, function(results)results[[1]])))
      subsample_disconcordant <- colSums(do.call(rbind, lapply(subsample_gamma_test, function(results)results[[2]])))
      return(mapply(FUN = godKrusGamma, subsample_concordant, subsample_disconcordant, kendallDenominator*100))
    })
    subsampled_gamma_CI <- apply(do.call(rbind,subsampled_gamma_cor),2,function(c)quantile(c,probs = c(0.025,0.975)))
    message("- Starting gamma rank permutation analysis, this could take a while...")
    nRandomObservations <- 10
    random_gamma_cor <- functToApply(seq_len(permutations), function(x){
      # Call gamma rank correlation
      null_gamma_test <- lapply(seq_len(nRandomObservations),function(x){
        cellToGamma(covariable, groups, labelOrder, indexes, cellCrowd, rank_index)})
      # Obtain the results and summarize
      random_concordant <- colSums(do.call(rbind,lapply(null_gamma_test,function(results)results[[1]])))
      random_disconcordant <- colSums(do.call(rbind,lapply(null_gamma_test,function(results)results[[2]])))
      # Goodman and Kruskal's gamma Formula
      return(mapply(FUN = godKrusGamma, random_concordant, random_disconcordant, kendallDenominator*nRandomObservations))
    })
    random_gamma_cor <- do.call(rbind, random_gamma_cor)
    # calculate the p.value
    higuer_in_null <- round((rowSums(apply(random_gamma_cor[,indexes],1, function(x){original_gamma_cor <= x}), na.rm = TRUE)) / (permutations), digits=5)
    lower_in_null <- round((rowSums(apply(random_gamma_cor[,indexes],1, function(x){original_gamma_cor >= x}), na.rm = TRUE)) / (permutations), digits=5)
    p.vals <- apply(rbind(original_gamma_cor, higuer_in_null, lower_in_null), 2, function(x)ifelse(x[1]>0, x[2], x[3]))
    p.adj <- round(p.adjust(p=p.vals, method="bonferroni"), digits=5)
    table.results <- data.frame(groupGammaCor=round(original_gamma_cor, 4), round(t(contig_tab)[,labelOrder],3), p.adj, CI95low=round(subsampled_gamma_CI[1,indexes],4), CI95high=round(subsampled_gamma_CI[2,indexes],4))
    colnames(table.results) <- c("groupGammaCor", c(sapply(labelOrder,function(label)paste0("percent_in_",label))), "p.adj","CI95low", "CI95high")
    return(table.results)
  }else{
    #########################################
    # Fold-Change and Montecarlo Simulation #
    # Only 2 covariable model:
    # We will perform a Montecarlo Test to create a random distribution and compare it with the original differences.
    message("Only 2 groups detected.")
    message(paste("Computing Fold Change proportion over covariables for groups:",labelOrder[1],"vs",labelOrder[2]))
    if(!is.null(sample_id)){
      message(paste("Additional sub-level for testing:",sample_id))
      samples <- as.character(main_metadata[, sample_id])
      nPerSample <- table(data.frame(groups,samples))[labelOrder,]
      cellCrowd <- apply(nPerSample, 1, function(perCond){list(sqrt(perCond[perCond!=0]))})
      cellCrowd <- cellCrowd[labelOrder]
    }
    df <- data.frame(groups, covariable)
    df.table <- table(df)
    # Logit transformation:
    logit <- function(fractions){log(fractions/(1-fractions))}
    contig_tab <- t(apply(pseudoCount(df.table),1,function(row){row/sum(row)}))[labelOrder,]    # # CONTRUCTION
    #original_test <- log2(contig_tab[1,] / contig_tab[2,])
    original_test <- logit(contig_tab[1,]) - logit(contig_tab[2,])
    indexes <- names(original_test)
    if(is.null(sample_id)){
      cellCrowd <- round(sqrt(c(table(groups))))
      cellCrowd <- cellCrowd[labelOrder]
    }
    # We perform the Montecarlo test
    message("- Starting montecarlo simulation of fold changes")
    null_test <- functToApply(seq_len(round(sqrt(permutations))), function(nParallelInstances){
      null_test_sub <- lapply(seq_len(permutations/round(sqrt(permutations))), function(permutationsPerInstance){
        cellToMontecarlo(covariable, groups, labelOrder, indexes, cellCrowd)
        })
    })
    # Unpack results
    null_test_fcs <- do.call(rbind, lapply(null_test,function(l)do.call(rbind,lapply(l,function(results)results[[1]]))))
    null_test_real <- do.call(rbind, lapply(null_test,function(l)do.call(rbind,lapply(l,function(results)results[[2]]))))
    # test 1 #
    # Calculate how extreme our observed values are in comparison with the random distribution
    # We want to see if the FoldChanges on the random distribution are lower or higher than the observed FoldChange
    higuer_in_null <- (rowSums(apply(null_test_fcs[,indexes],1, function(x){original_test <= x}))+1) / (permutations+1)
    lower_in_null <- (rowSums(apply(null_test_fcs[,indexes],1, function(x){original_test >= x}))+1) / (permutations+1)
    p.vals <- apply(rbind(original_test, higuer_in_null, lower_in_null), 2, function(x)ifelse(x[1]>0, x[2], x[3]))
    p.adj <- round(p.adjust(p = p.vals,method = "fdr"), digits = 5)
    sd.montecarlo <- apply(null_test_fcs, 2, sd)
    # test 2 #
    # We calculate the Coefficient Intervals for the Montecarlo simulation of the real data:
    intervals <- apply(null_test_real[,indexes], 2, function(c)quantile(c,probs = c(0.025,0.975)))
    table.results <- data.frame(groupFC=original_test, round(contig_tab[labelOrder[1],],3), round(contig_tab[labelOrder[2],],3), p.adj, sd.montecarlo, CI95low=intervals[1,], CI95high=intervals[2,])
    colnames(table.results) <- c("groupFC", paste0("percent_in_",labelOrder[1]), paste0("percent_in_",labelOrder[2]), "p.adj", "sd.montecarlo", "CI95low", "CI95high")
    # We make sure that CIs are lower or higuer than the observed value, if not make them equal:
    indx <- which(!(table.results[,"CI95low"] < table.results[,"groupFC"]))
    table.results[indx,"CI95low"] <- table.results[indx,"groupFC"]
    indx <- which(!(table.results[,"CI95high"] > table.results[,"groupFC"]))
    table.results[indx,"CI95low"] <- table.results[indx,"groupFC"]
    methods::show(plotAbundanceTest(tableResults=table.results, subtype_variable=subtype_variable))
    return(table.results)
  }
}
# Data simulation with 3 or > groups
# groups1 <- c(rep("CellA",700),rep("CellB",300),rep("CellC",500),rep("CellD",1000))
# groups2 <- c(rep("CellA",1700),rep("CellB",350),rep("CellC",550),rep("CellD",800))
# groups3 <- c(rep("CellA",1200),rep("CellB",200),rep("CellC",420),rep("CellD",800))
# groups4 <- c(rep("CellA",500),rep("CellB",1000),rep("CellC",10),rep("CellD",1200))
# groups <- c(rep("A",length(groups1)),rep("B",length(groups2)),rep("C",length(groups3)),rep("D",length(groups4)))
# labelOrder <- c("C","B","A","D")
# labelOrder <- c("B","D")
# covariable <- c(groups1, groups2,groups3,groups4)
# meta.data <- data.frame(groups, covariable)
# rownames(meta.data) <- as.character(1:nrow(meta.data))
# system.time(
# results <- lotOfCell(scObject = meta.data,
#                      main_variable = "groups",
#                      subtype_variable = "covariable",
#                      labelOrder = labelOrder,
#                      permutations = 1000)
#   )
# using do.call rbind:
# - Starting gamma rank permutation analysis, this could take a while...
# user  system elapsed
# 142.981   0.766 144.658

