%% LibSVMinGuassKernel.m
%
% THU2016 Foundations of Machine Learning assignment4 programming.
% Copyright Leakey, 20161119-20161122.
%
% Apply LibSVM in Guassian Kernel Support Vector Machine.
%
% Ref: NYU Machine Learning homework2.
% The original data can be downloaded from:
% http://www.cs.nyu.edu/~mohri/ml05/positive.dat
% http://www.cs.nyu.edu/~mohri/ml05/negative.dat
%
% This script needs Libsvm & SVMcgForClass.
%

%% Generating the data
SIGMA = 50 * eye(50);
mu1 = ones(1, 50);
mu2 = -1 * ones(1, 50);

positive_inst = mvnrnd(mu1, SIGMA, 5000);
positive_label = ones(5000, 1);
negative_inst = mvnrnd(mu2, SIGMA, 5000);
negative_label = -1 * ones(5000, 1);

libsvmwrite('Positive.txt', positive_label, sparse(positive_inst));
libsvmwrite('Negative.txt', negative_label, sparse(negative_inst));

%% Select Gaussian Kernel find c/g parametre
[positive_label, positive_inst] = libsvmread('positive.txt'); % read data
[negative_label, negative_inst] = libsvmread('negative.txt');
data_label = [positive_label; negative_label];
data_inst = [positive_inst; negative_inst];

rng(6); % random seed once a time for random function
instid = randperm(10000); % ref: http://www.ilovematlab.cn/thread-55405-2-1.html
data5k_inst = data_inst(instid(1:5000), :);% get 5k examples
data5k_label = data_label(instid(1:5000), :);
clearvars -except data5k*

%%

path = pwd;
mkdir(path, 'Figures');
Ntrain = 1000;%% Guass Kernel -- training set: 2000
rng(Ntrain);
instid1 = randperm(5000);
train_inst1 = data5k_inst(instid1(1:Ntrain), :);
train_label1 = data5k_label(instid1(1:Ntrain), :);
test_inst1 = data5k_inst(instid1(Ntrain+1:5000), :);
test_label1 = data5k_label(instid1(Ntrain+1:5000), :); % get training set
% choose -t 1 polynomial; -t 2 Guassian(RBF) -- radial basis function
% model1 = svmtrain(train_label1, train_inst1, '-t 2 -g 0.0001 -b 1 -c 1.678'); % train and test model 
% model1 = svmtrain(train_label1, train_inst1, '-t 2 -g 0.1 -b 1 -c 1');
[predict_label1, accuracy1, dec_values1] = svmpredict(test_label1, test_inst1, model1, '-b 1');
SVMcgForClass(train_label1, train_inst1, -1, 3, -17, -15, 3, 0.3, 0.1, 1.5)

saveas(gcf, [path, 'Figures', num2str(1000), '.jpg'])

%% Guass Kernel -- training set: 2000
Ntrain = 2000;
rng(Ntrain);
instid2 = randperm(5000);
train_inst2 = data5k_inst(instid2(1:Ntrain), :);
train_label2 = data5k_label(instid2(1:Ntrain), :);
test_inst2 = data5k_inst(instid2(Ntrain+1:5000), :);
test_label2 = data5k_label(instid2(Ntrain+1:5000), :);
model2 = svmtrain(train_label2, train_inst2, '-t 2 -g 0.0001 -b 1 -c 1.678'); 
[predict_label2, accuracy2, dec_values2] = svmpredict(test_label2, test_inst2, model2, '-b 1');
SVMcgForClass(train_label2, train_inst2, -1, 3, -17, -15, 3, 0.5, 0.2, 1.5);

%% Guass Kernel -- training set: 3000
Ntrain = 3000;
rng(Ntrain);
instid3 = randperm(5000);
train_inst3 = data5k_inst(instid3(1:Ntrain), :);
train_label3 = data5k_label(instid3(1:Ntrain), :);
test_inst3 = data5k_inst(instid3(Ntrain+1:5000), :);
test_label3 = data5k_label(instid3(Ntrain+1:5000), :);
model3 = svmtrain(train_label3, train_inst3, '-t 2 -g 0.0001 -b 1 -c 1.678'); 
[predict_label3, accuracy3, dec_values3] = svmpredict(test_label3, test_inst3, model3, '-b 1');
SVMcgForClass(train_label3, train_inst3, -1, 3, -17, -15, 3, 0.5, 0.2, 1.5);

%% Guass Kernel -- training set: 4000
Ntrain = 4000;
rng(Ntrain);
instid4 = randperm(5000);
train_inst4 = data5k_inst(instid4(1:Ntrain), :);
train_label4 = data5k_label(instid4(1:Ntrain), :);
test_inst4 = data5k_inst(instid4(Ntrain+1:5000), :);
test_label4 = data5k_label(instid4(Ntrain+1:5000), :);
model4 = svmtrain(train_label4, train_inst4, '-t 2 -g 0.0001 -b 1 -c 1.678'); 
[predict_label4, accuracy4, dec_values4] = svmpredict(test_label4, test_inst4, model4, '-b 1');
SVMcgForClass(train_label4, train_inst4, -5, 3, -20, -5, 3, 1, 1, 1.5);

%% Guass Kernel -- training set: 5000
% use all the training set as test
Ntrain = 5000;
rng(Ntrain);
instid5 = randperm(5000);
train_inst5 = data5k_inst(instid5(1:Ntrain), :);
train_label5 = data5k_label(instid5(1:Ntrain), :);

model5 = svmtrain(train_label5, train_inst5, '-t 2 -g 0.0001 -b 1 -c 1.678'); 
[predict_label5, accuracy5, dec_values5] = svmpredict(train_label5, train_inst5, model5, '-b 1');
SVMcgForClass(train_label4, train_inst4, -1, 3, -17, -15, 3, 0.5, 0.2, 1.5);

%% Compute the Bayes Error.
SIGMA = 50 * eye(50);
mu1 = ones(1, 50);
mu2 = -1 * ones(1, 50);
rho = (mu2 - mu1) * inv(SIGMA) * (mu2 - mu1)';

syms x
f = exp(-x^2)/sqrt(2*pi);
Pr_error = int(f, x, rho/2, inf);
Pr_error = vpa(Pr_error);  % calculate the integal.
% Here is an question: why .16% is lower bound, the result shows ~16% is lower bound 
