#!/usr/bin/env Rscript
args <- commandArgs(TRUE)


library(bnlearn, quietly = T)
library(parallel, quietly = T)
library(graph, quietly = T)
library(igraph, quietly = T)

# Litte string concatenation helper function
concat<- function(..., sep='') { paste(..., sep=sep, collapse=sep) }

core <- detectCores()
verbose <- FALSE
max_level <- as.integer(args[2])

dataset_name <- args[1]

print(concat("Executing for ", dataset_name, " with maximum level of ", max_level, "."))

df <- read.csv(concat("../data/", dataset_name, "/", dataset_name, "_encoded.csv"), header=FALSE, check.names=TRUE, sep=",")
df[] <- lapply(df, factor)
df <- df[sapply(df, function(x) !is.factor(x) | nlevels(x) > 1)]
matrix_df <- df

cl = makeCluster(core, type = "PSOCK")
result = pc.stable(matrix_df, debug=verbose, test="x2" , alpha=0.05, max.sx=max_level, cluster=cl)
stopCluster(cl)

print(result)

graph_i <- graph_from_graphnel(as.graphNEL(skeleton(result)))
print(graph_i)
write_graph(graph_i, concat("../data/", dataset_name, "/bnlearn_" , dataset_name, "_graph_max_level_", max_level, ".gml", sep=""), "gml")
