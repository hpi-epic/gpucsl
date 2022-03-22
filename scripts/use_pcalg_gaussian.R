#!/usr/bin/env Rscript
args <- commandArgs(TRUE)

library(pcalg)
library(graph)
library("XML")
library(igraph)
library(here)

# Litte string concatenation helper function
concat<- function(..., sep='') { paste(..., sep=sep, collapse=sep) }

dataset_name <- args[1]
print(concat("Executing for dataset: ", dataset_name))

# read data
dataset_path <- file.path(concat(here(), "/data/", dataset_name, "/", dataset_name,  ".csv"), fsep=.Platform$file.sep)
dataset <- read.table(dataset_path, sep=",")

# Prepare data
corrolationMatrix <- cor(dataset)
p <- ncol(dataset)
suffStat <- list(C = corrolationMatrix, n = nrow(dataset))

stable_fast_fit <- pc(suffStat, indepTest=gaussCItest, p=p, skel.method="stable.fast", alpha=0.05, NAdelete=TRUE)

output_path <- concat(here(), "/data/", dataset_name, "/pcalg_", dataset_name)

write(stable_fast_fit@zMin, file(paste(output_path, "_zMin.txt", sep="")))
write(stable_fast_fit@max.ord, file(paste(output_path, "_max.ord.txt", sep="")))
write(stable_fast_fit@n.edgetests, file(paste(output_path, "_n.edgetests.txt", sep="")))
write.table(format(stable_fast_fit@pMax, digits=2), file(paste(output_path, "_pMax.csv", sep="")),sep=",",row.names=FALSE, col.names=FALSE)

# Outputs a really simple sepset format: node_one node_two length(sepset) (...members of sepset)
sink(concat(output_path, "_sepset.txt", sep=""))
length <- length(stable_fast_fit@sepset)
for(i in 1:length) {
	seplist <- stable_fast_fit@sepset[[i]]
	for(j in 1:length(seplist)) {
		sepset = seplist[[j]]
		if (length(sepset) > 0) {
			cat(sprintf("%d %d %d ",i, j, length(sepset)))
			cat(sepset, "\n")
		}
	}
}

sink()

# https://igraph.org/r/doc/graph_from_graphnel.html
graph_i <- graph_from_graphnel(stable_fast_fit@graph, weight=FALSE)
# https://igraph.org/r/doc/write_graph.html
write_graph(graph_i, paste(output_path, "_graph.gml", sep=""), "gml")

