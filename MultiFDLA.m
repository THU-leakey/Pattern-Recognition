%% Multi Fisher Discriminant Analysis
% Copyright(C) Li Ji. 20160409 - 20160413. 
% Multiple Fisher's linear discriminant analysis
% Pre Step:
% PCA: Reduce the feature space M to the d dimentions(d is better not too
% large), that is to say, d <= nTrain - nGroups
% Then: Reduce the feature space d to the c-1 dimentions(4096=d >= c=40)
% Step1: 
% Calculate Sw, Sb under multiclasses(c classes)
% Step2:
% calculate the eigen value
% Step3:
% calculate the projection matrix W
% Step4:
% Project both the training set and the test samples to the c-1 dimentions
% space, and then use KNN(euclide or some other distances) or SVM to decide
% which class it is.

% Hint: Try 10 of 40 at first.

% The process of reducing the dimentional space:
% 4096 -> nTrain - nGroups -> nGroups - 1
% Then use KNN or SVM to classify the image

% Reference: 
%    PCA part & Calculate the eigenvector: LDA.m from: 
%    http://download.csdn.net/index.php/source/do_download/4088830/emiyasstar__/e5fec49b0c7f64e4af811facc0e4441e
%    updated by Emiyassstar(emiyassatr@foxmail.com),Feb,2012
%    Written by Deng Cai (dengcai@gmail.com), April/2004, Feb/2006

%% Step1: Load Data
clear all;
clc;
orl = load('orl_faces.mat'); % load orl face database
K = 11; % How many samples to be trained for each class: K
nGroups = 41; % How many classes to be calculated: nGroups
while(K >= 10||nGroups > 40) % User input: 
    nGroups = input('Input number of groups(0 < N <= 40, integer):');
    K = input('Input number of train sample in each group(0 < K < 10, integer): ');
end
nSamples = 10*nGroups; % sample numbers
label = orl.label(1:nSamples, :);
data = orl.data(1:nSamples, :);
clear orl;
[N, M]      = size(data);
nTrain      = nGroups*K; % number of train set
nTest       = nSamples - nTrain; % number of test set
train       = zeros(nTrain, M);
trntag      = zeros(nTrain, 1);
test        = zeros(nTest, M);
tsttag      = zeros(nTest, 1);
clear nSamples;
trnj = 1;tstj = 1;
for i = 1:N
    if(mod(i, 10) < K)
        train(trnj, :) = data(i, :);
        trntag(trnj) = label(i);
        trnj = trnj + 1;
    else
        test(tstj, :) = data(i, :);
        tsttag(tstj) = label(i);
        tstj = tstj + 1;
    end
end
clear i trnj tstj N label data;

%% Step2: PCA Reduce the Feature Dimention
d = nTrain - nGroups;
if M > d
    [dataset_coef, train_score,~,~] = pca(train);
    test_score = bsxfun(@minus, test, mean(train, 1)) * dataset_coef;
    train = train_score(:, 1:d); % reduce down to nTrain - nGroups dimention
    test = test_score(:, 1:d);
    [~,M] = size(train);
end
clear test_score dataset_coef train_score;

%% Step3: Multi-Fisher
% calculate the mean of every class
Miall = zeros(nGroups, M); % mean Matrix
Swiall = zeros(nGroups, M, M); 
for iGroups = 1:nGroups
    Lsrt = K * (iGroups - 1) + 1;
    Lend = K * iGroups;
    Miall(iGroups, :) = mean(train(Lsrt:Lend, :));
    Swiall(iGroups, :, :) = cov(train(Lsrt:Lend, :)) * (K-1);
end
Sw = reshape(sum(Swiall(1:nGroups, :, :)), M, M); %within-class cov matrix
clear Swiall;
m = mean(train(1:nTrain, :)); % the mean of total dataset
Sb = zeros(M, M); 
for iGroups = 1:nGroups
   temp =  Miall(iGroups, :) - m;
   Sb = Sb + K * ((temp') * temp);
end
% the aim is to get the largest real vector
% Sb*W = Sw * W * D; W is the eigen matrix d*(c-1)
[eigvector, eigvalue] = eigs((Sw\Sb), (nGroups - 1), 'LR');
% the eigen vector; wi is the col of it, i = 1, 2 ...,c-1

%% Step4: Project
% project the d dimention train and test samples to 1 dimention
trainLda = train * eigvector; % reduce down to the nGroups - 1 dimention
testLda = test * eigvector;

%% Step5: Classify by KNN
%[idx, ~] = knnsearch(trainLda, testLda, 'dist', 'euclidean', 'K', K);%94%
% try different distance get similar output
%[idx, ~] = knnsearch(trainLda, testLda, 'dist', 'seuclidean', 'K', K);%91%
[idx, ~] = knnsearch(trainLda, testLda, 'dist', 'cityblock', 'K', K);%93.5
% convert idx to the testtag
testtag = zeros(nTest, K);
for i = 1:K
    for j = 1:nTest
        testtag(j, i) = trntag(idx(j, i));
    end
end



%记录所有结果，然后每一个相等的众数求加权和
 predtag = mode(testtag, 2);
 nCorrect = 0;
for i = 1:nTest
    if(predtag(i) == tsttag(i))
        nCorrect = nCorrect + 1;
    end
end
accurcy=strcat(num2str(nCorrect/nTest*100),'%')