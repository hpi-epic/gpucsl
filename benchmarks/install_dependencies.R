#!/usr/bin/env Rscript

# install to local user library path
install.packages("BiocManager", lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
# Bioconductor version (works for both Bioconductor and CRAN packages)
BiocManager::install(version = "3.14", update = FALSE, lib = Sys.getenv("R_LIBS_USER"))

# install.packages("BiocManager")
# BiocManager::install()
BiocManager::install(c("graph", "RBGL", "Rgraphviz"), update = FALSE, lib = Sys.getenv("R_LIBS_USER"))
install.packages("pcalg",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
install.packages("XML",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
install.packages("optparse",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
install.packages("tictoc",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
install.packages("here",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
install.packages("bnlearn",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
install.packages("igraph",lib = Sys.getenv("R_LIBS_USER"), repos = "https://cran.rstudio.com/")
