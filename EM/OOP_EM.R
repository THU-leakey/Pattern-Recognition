# Copyright(c) 2015/11/28 - 2015/11/30 Li Ji. All rights reserved.
# EM alogrithm tests OOPS model

# initialize the working enviroment
wd<-"F:/SI_Pro_Nov30/EM Alogrithm"
setwd(wd)

# define globle variables
N=100                                  #number of sequences,nrow(X),100 in this question
K=6                                    #length of motif in OOPS model
M=4                                    #number of nucleotides
L=30                                   #length of each sequence
dif=0.0000001                          #a slight diffrence between two likehood
 
#define matrices to be used 
x<-matrix(NA,nrow = N, ncol=L)         #record all the dna data
colnames(x)<-c(1:L)
theta0<-matrix(NA,nrow=M,ncol=1)       #background pwm theta0 matrix
row.names(theta0)<-c("A","C","G","T")
theta<-matrix(NA,nrow=M,ncol=K)        #foreground pwm theta matrix
row.names(theta)<-c("A","C","G","T")
colnames(theta)<-c(1:K)
y<-matrix(NA,nrow=N,ncol=(L-K+1))      #P(Xi|starting position=j,theta,theta0)
colnames(y)<-c(1:(L-K+1))
z<-matrix(NA,nrow=N,ncol=(L-K+1))      #normalization of y
colnames(z)<-c(1:(L-K+1))
NtotalACGT<-matrix(0,nrow=M,ncol=1)    #keep the total numbers of ACGT in 100 seqs
row.names(NtotalACGT)<-c("A","C","G","T")
colnames(NtotalACGT)<-c("TotalNumber")
NmotifACGT<-matrix(NA,nrow=M,ncol=K)   #save ACGT numbers in each position in motif
row.names(NmotifACGT)<-c("A","C","G","T")
colnames(NmotifACGT)<-c(1:K)
NbgACGT<-matrix(0,nrow=M, ncol=1)    #record the current background ACGT numbers
row.names(NbgACGT)<-c("A","C","G","T")
colnames(NbgACGT)<-c("background")
sumofrow<-matrix(0,nrow = N, ncol =1)  #record the sum of each row

#define functions to be used 
theta0pro<-function(A){
#function:get the current probablity of A/C/G/T from theta0 of the background dna
#parameter: A:current x[i,j] from background
  if (A=="A")
  {
    return(as.numeric(theta0[1,1]))
  }
  if (A=="C")
  {
    return(as.numeric(theta0[2,1]))
  }
  if (A=="G")
  {
    return(as.numeric(theta0[3,1]))
  }
  if(A=="T")
  {
    return(as.numeric(theta0[4,1]))
  }
}

thetakpro<-function(A,k){
#parameter:A:current x[i,j] from foreground;k:1,2...K,kth in motif
  if (A=="A")
  {
    return(as.numeric(theta[1,k]))
  }
  if (A=="C")
  {
    return(as.numeric(theta[2,k]))
  }
  if (A=="G")
  {
    return(as.numeric(theta[3,k]))
  }
  if(A=="T")
  {
    return(as.numeric(theta[4,k]))
  }
}


NumACGT<-function(A){
#calculate the number of A/C/G/T of the input dataset
#parameter:A is a col of 100 rows
  a=0;c=0;g=0;t=0
  for (i in 1:100)
  {
    if (A[i]=="A")
    {
      a=a+1;
    }
    if (A[i]=="C")
    {
      c=c+1;
    }
    if (A[i]=="G")
    {
      g=g+1;
    }
    if (A[i]=="T")
    {
      t=t+1;
    }
  }
  return (list(a,c,g,t))
}

###raw data process###
X<-read.table("sequences.txt", header=F, sep="", as.is=F, colClasses = "character") #get the raw data&make it a matrix
for(i in 1:N)
{
  temp1<-strsplit(X[i,],split="")  #save a row
  temp2<-unlist(temp1)             #unlist
  x[i,]<-matrix(temp2,ncol=L)
}
rm(temp1);rm(temp2);rm(X)

#get the total number of A/C/G/T in the 100 seqs
for(i in 1:N)
{
  for(j in 1:L)
  {
    if(x[i,j]=="A")
    {
      NtotalACGT[1,1]=NtotalACGT[1,1]+1
    }
    if(x[i,j]=="C")
    {
      NtotalACGT[2,1]=NtotalACGT[2,1]+1
    }
    if(x[i,j]=="G")
    {
      NtotalACGT[3,1]=NtotalACGT[3,1]+1
    }
    if(x[i,j]=="T")
    {
      NtotalACGT[4,1]=NtotalACGT[4,1]+1
    }
  }
}

#initialize the parameter matrices by random numbers
theta0[,1]<-runif(4)        #theta0 matrix:background
sumtemp<-sum(theta0[,1])    #sum of each column is 1
for(m in 1:M)
{
  theta0[m,1]=(theta0[m,1]/sumtemp)
}
for(k in 1:K)                #theta matrix:foreground
{
  theta[,k]<-runif(4)
  sumtemp<-sum(theta[,k])    #sum of each column is 1
  for(m in 1:M)
  {
    theta[m,k]=(theta[m,k]/sumtemp)
  }
}
theta0ini<-theta0
thetaini<-theta

###EM algrithm###
#initialize the variables
flag = 1             #iteration condition;1 continue iteration;0 stop iteration
n = 0                #iteration count
logLHPre = 0         #keep the log likelihood of the previous iteration

while(flag == 1)
{                  
#E-step:
#based on CURRENT:theta0,theta & KNOWN:x;
#calculate starting probability of every position of every row 
for (i in 1:N)
{
#the motif starts at the first column
pro=1
for (j in 1:K)# in the motif,foreground
{
  p1<-thetakpro(x[i,j],j)
  pro<-prod(pro,p1)
}
for (j in (1+K):L)
{
  p0<-theta0pro(x[i,j])
  pro<-prod(pro,p0)   #pro means probablity
} 
y[i,1]=pro#save the result in matrix y

#in the middle of sequence
for (p in 2:(L-K)) 
{
  pro=1
  for (j in 1:(p-1))#before motif,background
  {
    p0<-theta0pro(x[i,j])
    pro<-prod(pro,p0)   #pro means probablity
  } 
  for (j in p:(p+K-1))# in the motif,foreground
  {
    p1<-thetakpro(x[i,j],(j-p+1))
    pro<-prod(pro,p1)
  }
  for (j in (p+K):L)
  {
    p0<-theta0pro(x[i,j])
    pro<-prod(pro,p0)   #pro means probablity
  } 
  y[i,p]=pro
}

#the motif ends at the last col
pro=1
for (j in 1:(L-K))
{
  p0<-theta0pro(x[i,j])
  pro<-prod(pro,p0)              #pro means probablity
} 
for (j in (L-K+1):L)             # in the motif,foreground
{
  p1<-thetakpro(x[i,j],(j-L+K))
  pro<-prod(pro,p1)
}
y[i,(L-K+1)]=pro                       #save the result in matrix y
sumofrow[i,1]<-sum(y[i,])              #get the sum of ith row of y matrix
for(j in 1:(L-K+1))                    #normalize y matrix
{
  z[i,j]=(y[i,j]/sumofrow[i,1])
}
}

logLH<-log((1/(L-K+1)),base = exp(1))
for (i in 1:N)
{
  logLH <- logLH + log(sumofrow[i, 1],base = exp(1))  #loglikehood of complete data
}

#iteration condition judgement:
if(abs(logLH-logLHPre)<=dif)           #likehood NOT change too much
{
  flag=0
}
else
{
  logLHPre=logLH
}
n=n+1                                 #counting the iteration times

#M-step:Estimating theta & theta0
#sum over position 1 where A appears
for(k in 1:K)
{
  for(m in 1:M)
  {
    sumtemp=0
    for(i in 1:N)
    {
      for(j in 1:(L-K+1))
      {
        if(row.names(theta0)[m]==x[i,(j+k-1)])   #use the rownames of theta0 as an alphabet
        {
          sumtemp<-sumtemp+z[i,j]
        }
      }
    } 
    NmotifACGT[m,k]<-sumtemp
  }
}

#get the NbgACGT matrix
for (m in 1:M)
{
  NbgACGT[m,1]<-(NtotalACGT[m,1]-sum(NmotifACGT[m,]))
}
theta0pre<-theta0                      #keep theta0 matrix for    
sumtemp<-sum(NbgACGT[,1])+4            #add sum of col with turbulence in case result = 0
for (m in 1:M)                         #update theta0
{
  theta0[m,1]<-((NbgACGT[m,1]+1)/sumtemp)#get some interptance in case get zero
}
# update theta
thetapre<-theta         #keep theta matrix temporarily
for (k in 1:K)
{
  sumtemp<-sum(NmotifACGT[,k])+4 #get the sum and input interpence
  for (m in 1:M)
  {
   theta[m,k]<-((1+NmotifACGT[m,k])/sumtemp)
  }
}
}

rm(i, j, k, m, flag, p1, pro, sumtemp, p0, p)  # remove temp variables

