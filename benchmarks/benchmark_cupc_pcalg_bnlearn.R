library(graph)
library(MASS)
library(tictoc)
library(igraph)
library (optparse)
library(parallel)

concat<- function(..., sep='') { paste(..., sep=sep, collapse=sep) }


option_list <- list ( make_option (c("-d","--datasets_gaussian"),default="",
                                   help="list of datasets_gaussian"),
                      make_option (c("-c","--datasets_discrete"),default="",
                                   help="list of datasets_discrete"),
                      make_option (c("-o","--output_dir"),default="",
                                   help="output dir"),
                      make_option (c("-l","--max_level"),default="",
                                    help="max level"),
                      make_option (c("-b","--libraries"),default="",
                                    help="libraries to run")
                     )

parser <-OptionParser(option_list=option_list)
arguments <- parse_args (parser, positional_arguments=TRUE)
opt <- arguments$options
args <- arguments$args
datasets_gaussian <- gsub("\'", "", strsplit(opt$datasets_gaussian, ",")[[1]])
datasets_discrete <- gsub("\'", "", strsplit(opt$datasets_discrete, ",")[[1]])
datasets_gaussian <- datasets_gaussian[datasets_gaussian != ""]
datasets_discrete <- datasets_discrete[datasets_discrete != ""]
output_dir <- opt$output_dir
max_level <- opt$max_level
libraries <- opt$libraries
alpha <- 0.05
print(datasets_discrete)
print(length(datasets_discrete))
print(datasets_gaussian)
print(length(datasets_gaussian))

print(output_dir)
print(max_level)

cupc_results <- matrix(NA, nrow=length(datasets_gaussian), ncol=7)
colnames(cupc_results) <- c("library", "dataset", "distribution", "full_runtime", "discover_skeleton_time", "edge_orientation_time", "kernel_time")

pcalg_results <- matrix(NA, nrow=length(datasets_gaussian), ncol=7)
colnames(pcalg_results) <- c("library", "dataset", "distribution", "full_runtime", "discover_skeleton_time", "edge_orientation_time", "kernel_time")

bnlearn_results <- matrix(NA, nrow=length(datasets_discrete), ncol=7)
colnames(bnlearn_results) <- c("library", "dataset", "distribution", "full_runtime", "discover_skeleton_time", "edge_orientation_time", "kernel_time")

cores = detectCores()
print(paste("MAX CORES", cores))

skelmethod <- "stable.fast"

use_bnlearn <- grepl("bnlearn", libraries, fixed = TRUE)
use_pcalg <- grepl("pcalg", libraries, fixed = TRUE)
use_cupc <- grepl("cupc", libraries, fixed = TRUE)

for (i in seq_len(length(datasets_discrete))) {
  dataset_name = datasets_discrete[i]
  print("Discrete dataset")
  print(dataset_name)
  print(i)

  dataset_path <- concat("../../data/", dataset_name, "/", dataset_name, "_encoded.csv")

  if (use_bnlearn) {

    library(bnlearn, quietly = T)
    # BNLEARN setup
    df <- read.csv(dataset_path, header=FALSE, check.names=TRUE, sep=",")
    verbose <- FALSE
    max_level <- 3
    df[] <- lapply(df, factor)
    df <- df[sapply(df, function(x) !is.factor(x) | nlevels(x) > 1)]
    matrix_df <- df

    print(paste("running bnlearn on ", dataset_name))
    start.time <- Sys.time()
    # BNLEARN CALL

    cl = makeCluster(cores, type = "PSOCK")
    result = pc.stable(matrix_df, debug=verbose, test="x2" , alpha=alpha, max.sx=max_level, cluster=cl)
    stopCluster(cl)

    end.time <- Sys.time()
    time.taken <- end.time - start.time
    print(paste("the total time consumed by bnlearn on ", dataset_name))
    print(time.taken)

    print(result)
    bnlearn_results[i,] <- c(paste("bnlearn-maxlevel=", max_level, "-core=", cores, sep=""), dataset_name, "discrete", as.numeric(time.taken, units="secs"), as.numeric(time.taken, units="secs"), 0.0, 0.0)

  }

}

for (i in seq_len(length(datasets_gaussian))) {
  dataset_name = datasets_gaussian[i]
  print("Gaussian dataset")
  print(dataset_name)
  # read data
  dataset_path <- file.path(paste("../../data/", dataset_name, "/", dataset_name, ".csv", sep=""), fsep=.Platform$file.sep)
  dataset <- read.table(dataset_path, sep=",")

  # Prepare data
  p <- ncol(dataset)
  correlationMatrix <- cor(dataset)
  suffStat <- list(C = correlationMatrix, n = nrow(dataset))

  if (use_pcalg) {
    library(pcalg)

    print(paste("running pcalg ", skelmethod, " on ", dataset_name))
    start.time <- Sys.time()

    stable_fast_fit <- pc(suffStat, indepTest=gaussCItest, p=p, numCores = cores, skel.method=skelmethod, alpha=alpha, NAdelete=TRUE, m.max=max_level)

    end.time <- Sys.time()
    time.taken <- end.time - start.time

    print(paste("the total time consumed by pcalg on ", dataset_name))
    print(time.taken)
    print(stable_fast_fit)
    pcalg_results[i,] <- c(paste("pcalg-", skelmethod, cores, sep=""), dataset_name, "gaussian", as.numeric(time.taken, units="secs"), as.numeric(time.taken, units="secs"), 0.0, 0.0)
  }


  if (use_cupc) {
    source("cuPC.R")

    print(paste("running cupc on ", dataset_name))
    start.time <- Sys.time()

    cuPC_fit <- cu_pc(suffStat, gaussCItest, p=p, alpha=alpha, m.max=max_level, skel.method=skelmethod)

    end.time <- Sys.time()
    time.taken <- end.time - start.time

    print(paste("the total time consumed by cupc on ", dataset_name))
    print(time.taken)
    print(cuPC_fit)
    cupc_results[i,] <- c("cupc", dataset_name, "gaussian", as.numeric(time.taken, units="secs"), cuPC_fit$skeltime, cuPC_fit$edgetime, cuPC_fit$kernel_time)
  }

}


if (use_cupc) {
  write.csv(cupc_results, file = paste(output_dir, "/cupc.csv", sep=""))
}
if (use_pcalg) {
  write.csv(pcalg_results, file = paste(output_dir, "/pcalg.csv", sep=""))
}
if (use_bnlearn) {
  write.csv(bnlearn_results, file = paste(output_dir, "/bnlearn.csv", sep=""))
}
