#!/usr/bin/env Rscript
args <- commandArgs(TRUE)

library(dagitty)
library(igraph)
library(SEMgraph)

concat<- function(..., sep='') { paste(..., sep=sep, collapse=sep) }


library(here)

dataset_name <- args[1]
print("CURRENT FILE LOCATION")
print(here())
folder_path <- concat(here(), "/data/", dataset_name, "/")
num_nodes <- 100

gpucsl_graph <- read.graph(concat(folder_path, "output.gml"),format=c("gml"))
ground_truth_graph <- read.graph(concat(folder_path, "ground_truth.gml"),format=c("gml"))
pcalg_graph <- read.graph(concat(folder_path, "pcalg_", dataset_name, "_graph.gml"),format=c("gml"))

print(ground_truth_graph)
print(gpucsl_graph)
print(pcalg_graph)

d_ground_truth_graph <- graph2dagitty(ground_truth_graph %>% set_vertex_attr("name", value = seq(0,num_nodes)))
d_gpucsl_graph <- graph2dagitty(gpucsl_graph  %>% set_vertex_attr("name", value = seq(0,num_nodes)))
d_pcalg_graph <- graph2dagitty(pcalg_graph  %>% set_vertex_attr("name", value = seq(0,num_nodes)))


e1 <- equivalenceClass(d_ground_truth_graph)
e2 <- equivalenceClass(d_gpucsl_graph)
e3 <- equivalenceClass(d_pcalg_graph)

print("==== ground_truth")
print(e1)

print("==== gpucsl")
print(e2)

print("==== pcalg")
print(e3)


v1 <- unlist(strsplit(toString(e1),"\n"))
v1e <- sort(v1[grepl(glob2rx("*-*"), v1)])

v2 <- unlist(strsplit(toString(e2),"\n"))
v2e <- sort(v2[grepl(glob2rx("*-*"), v2)])

v3 <- unlist(strsplit(toString(e3),"\n"))
v3e <- sort(v3[grepl(glob2rx("*-*"), v3)])


n <- max(length(v1e), length(v2e), length(v3e))
length(v1e) <- n
length(v2e) <- n
length(v3e) <- n

print(cbind(v1e, v2e, v3e))

plot(graphLayout(e1))
plot(graphLayout(e2))
plot(graphLayout(e3))


print("eq(GROUND_TRUTH) == eq(GPUCSL)")
print(e1 == e2)

print("eq(GROUND_TRUTH) == eq(PCALG)")
print(e1 == e3)

print("eq(PCALG) == eq(GPUCSL)")
print(e3 == e2)

