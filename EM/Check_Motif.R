# Copyright(c) 2015/11/30 Li Ji. All rights reserved.
# check whether the two types of motif are the same

N <- 100
L <- 6
wd <- "F:/SI_Pro_Nov30/EM Alogrithm/checkmotif"
motifile1 <- "19 th result motif.txt"
motifile2 <- "3 th result motif.txt"
setwd(wd)

x1 <- matrix(NA,nrow = N, ncol = L)  # record one motif file data
colnames(x1) <- c(1:L)
x2 <- matrix(NA,nrow = N, ncol = L)  # record another motif file data
colnames(x2) <- c(1:L)

X<-read.table(motifile1, header = F, sep = "", as.is = F, colClasses = "character")  # get the raw data&make it a matrix
for(i in 1 : N){
  temp1 <- strsplit(X[i,],split = "")  # save a row
  temp2 <- unlist(temp1)  # unlist
  x1[i,] <- matrix(temp2,ncol=L)
}
rm(temp1, temp2, X)
X<-read.table(motifile2, header = F, sep = "", as.is = F, colClasses = "character")  # get the raw data&make it a matrix
for(i in 1 : N){
  temp1 <- strsplit(X[i,],split = "")  # save a row
  temp2 <- unlist(temp1)  # unlist
  x2[i,] <- matrix(temp2,ncol=L)
}
rm(temp1, temp2, X)

for(i in 1 : N){
  for(k in 1 : L){
    if(x1[i, k] != x2[i, k])
      print(list(i,k))
  }
}
