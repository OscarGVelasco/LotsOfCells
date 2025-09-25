#' Calculate fold-changes from a random distribution of the given data.
#'
#' `cellToMontecarlo()` returns the fold-change from the data generated from the random sampling of the data provided.
#'
#' This function will perform a Montecarlo simulation from the groups and covariables provided. It will compute the fold-change from this generated distribution.
#'
#' @param covariable Vector. Labels corresponding with the covariable.
#' @param groups Vector. Labels corresponding with the main group variable.
#' @param labelOrder Vector. The labels in groups in the order of the desired comparison.
#' @param indexes Vector. Order of the covariable as in the original data.
#' @param cellCrowd Vector. Number of elements to be subsampled from each group.
#' @return A list of the computated fold-changes from the random distribution created using a montecarlo test.
#'
#' @author Oscar Gonzalez-Velasco
#' @keywords internal
cellToMontecarlo <- function(covariable=NULL,groups=NULL, labelOrder, indexes, cellCrowd){
  # test 1 : the random distribution
  # We mix all the cells together and we subset at random sub-samples in proportion to the original samples
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
  pseudoCount <- function(counts){counts + sqrt((counts*counts)+1)}
  # Logit transformation:
  logit <- function(fractions){log(fractions/(1-fractions))}
  # Obtain Frequencies of classes
  contig_tab <- t(apply(pseudoCount(dftmp),2,function(row){row/(sum(row)+1)}))[labelOrder, indexes]

  # test 2 : We recreate the original samples by picking random subsamples of the same samples, without mixing
  # The idea is to create a normal distribution from the original data, and gain some information about coeficient intervals of the calculated FC
  if(is.list(cellCrowd)){
    cond1 <- lapply(cellCrowd[[1]][[1]], FUN = function(nCellToSample){sample(covariable[groups %in% labelOrder[1]], size = nCellToSample, replace = T)})
    cond1 <- unname(unlist(cond1))
    cond2 <- lapply(cellCrowd[[2]][[1]], FUN = function(nCellToSample){sample(covariable[groups %in% labelOrder[2]], size = nCellToSample, replace = T)})
    cond2 <- unname(unlist(cond2))
    dforigin <- data.frame(covariable=c(cond1, cond2),
                        groups = c(rep(labelOrder[1], times=length(cond1)), rep(labelOrder[2], times=length(cond2))))

  }else{
    dforigin <- data.frame(covariable=c(sample(covariable[groups %in% labelOrder[1]],size = cellCrowd[labelOrder[1]], replace = T),
                            sample(covariable[groups %in% labelOrder[2]],size = cellCrowd[labelOrder[2]], replace = T)),
                groups = c(rep(labelOrder[1],cellCrowd[labelOrder[1]]),rep(labelOrder[2],cellCrowd[labelOrder[2]])))
  }
  dforigin <- table(dforigin)
  if(!all(indexes %in% rownames(dforigin))){
    tmp <- do.call(rbind, rep(list(rep(0,ncol(dforigin))), sum(!(indexes %in% rownames(dforigin)))))
    rownames(tmp) <- indexes[!(indexes %in% rownames(dforigin))]
    dforigin <- rbind(dforigin, tmp)[indexes,]
  }
  # dforigin[dforigin == 0] = 1 * instead of using pseudocount 1 use arcsin:
  pseudoCount <- function(counts){counts + sqrt((counts*counts)+1)}
  contig_tab_origin <- t(apply(pseudoCount(dforigin),2,function(row){row/(sum(row)+1)}))[labelOrder, indexes]

  #return(list(log2(contig_tab[1,] / contig_tab[2,]), log2(contig_tab_origin[1,] / contig_tab_origin[2,])))
  return(list(logit(contig_tab[1,]) - logit(contig_tab[2,]), logit(contig_tab_origin[1,]) - logit(contig_tab_origin[2,])))
}
