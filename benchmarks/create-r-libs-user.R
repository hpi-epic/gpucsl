#!/usr/bin/env Rscript

# for some reason this command has to be in its own file, otherwise the dependency installation via 
# R scripts executed from the command line fails
dir.create(path = Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)