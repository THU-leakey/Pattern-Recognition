# Copyright(c) 2015/11/30 Li Ji. All rights reserved.
# Find the max log likelyhood and find the most likely motifs in sequences

# initialize the working enviroment
rm(list = ls())                          
wd<-"F:/SI_Pro_Nov30/EM Alogrithm"
setwd(wd)

# define globle variables
N <- 100                                  #number of sequences,nrow(X),100 in this question
K <- 6                                    #length of motif in OOPS model
M <- 4                                    #number of nucleotides
L <- 30                                   #length of each sequence

# define global variable
I <- 100  # EM times
# define the matrix
rl.matrix <- matrix(0, nrow = I, ncol = 1)  # save result
mll.matrix <- matrix(0, nrow = N, ncol = (1 + K))  # save most likely location of the motif
colnames(mll.matrix) <- c(1 : 6, "most likely position")

# keep the max of Q function data
for(w in 1 : I){
  source('OOP_EM.R')
  rl.matrix[w, 1] <- logLH  # keep the logLH of ith result in the ith row
  save.image(file = (paste(w, " th result", ".RData", sep = "")))
}

# find the best result from all the data
max.loci <- which.max(rl.matrix[, 1])
load(file = (paste(max.loci, " th result", ".RData", sep = "")))
for (i in 1 : N){
  temp <- which.max(z[i, ])
  mll.matrix[i, 7] <- temp
  for (j in temp : (temp + K -1)){
    mll.matrix[i, (j - temp + 1)] <- x[i, j]
  }
}
write.table(mll.matrix[1 : 100, 1 : 6], file = (paste(max.loci, " th result motif", ".txt", sep = "")), quote = FALSE, sep = "", row.names = FALSE, col.names = FALSE)

