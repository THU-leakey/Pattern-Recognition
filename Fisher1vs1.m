%% Fisher Linear Discriminant Analysis: 1 vs 1
% Copyright(C) Li Ji. 20160409 - 20160413
%
% Fisher's linear discriminant with "1 vs 1"
%
% 1 vs 1 strategy: 
% Step1: 
% Implement fisher linear discriminant on every 2 classes, e.g. Ci & Cj;
% Step2:
% There  are 40 classes in all. This strategy will lead to C(2,40)=780
% binary discriminant functions in total. 
% Step3:
% Each point is then classified according to a majority vote amongst the 
% discriminant functions. By the largest number of the total test.
% Hint: Try 10 of 40 at first.

% Error rate is very high
%% Pre Step: Load Data
orl = load('orl_faces.mat'); % load orl face database
K = 11; % How many samples to be trained for each class: K
nGroups = 41; % How many classes to be calculated: nGroups
while(K >= 10||nGroups > 40) % User input: 
    nGroups = input('Input number of groups(3 < N <= 40, integer):');
    K = input('Input number of train sample in each group(3 < K < 10, integer): ');
end
nSamples = 10*nGroups; % sample numbers
label = orl.label(1:nSamples, :);
data = orl.data(1:nSamples, :);
[N, M]      = size(data);
clear orl;

%% Step 1: PCA
d = 2*(K - 1); % set the dimentions to reduce
if M > d
    [~, train_score, latent] = pca(zscore(data));
    data = train_score(:, 1:d); % reduce down to nTrain - nGroups dimention
    [~,M] = size(data);
end
clear train_score;

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
clear i trnj tstj N ;

index = nchoosek(1:nGroups, 2);  % get the index of Fisher 1 vs 1
nClass = length(index);
testtag = zeros(nTest, nClass); % record the test tag

%% Step2: Fisher 1 vs 1
% get the training target of fisher by the Classifier
for nClassier   = 1:nClass
    L1srt       = K*(index(nClassier, 1) - 1) + 1;
    L1end       = L1srt + K - 1;
    L2srt       = K*(index(nClassier, 2) - 1) + 1;
    L2end       = L2srt + K - 1;
    fitrntag    = [trntag(L1srt:L1end);trntag(L2srt:L2end)];
    
    % get the training sample of fisher by the training target
    C           = unique(fitrntag);
    fitrntagC1  = find(trntag == C(1));
    fitrntagC2  = find(trntag == C(2));
    fitrnC1     = train(fitrntagC1, :);
    fitrnC2     = train(fitrntagC2, :);
    fitrn       = [fitrnC1; fitrnC2];
    
    % calculate the mean vector
    m1 = mean(fitrnC1);
    m2 = mean(fitrnC2);
    
    % with-inclass separate scatter
    S1 = cov(fitrnC1) *  (K - 1);
    S2 = cov(fitrnC2) *  (K - 1);
    Sw = S1 + S2;
    w = Sw\((m1 - m2)'); 
    % warning('') here：
    % 警告: 矩阵接近奇异值，或者缩放错误。结果可能不准确。RCOND =  1.186869e-23。
    % When (sample)N > (character)M, Sw is NOT singular, but here N << M
    
    %% Step 3: Project & Classify
    y = fitrn * w; % use current classifier
    testy = test(1:nTest, :) * w;
    k = 5;
    [idx, distance] = knnsearch(y, testy, 'dist', 'euclidean', 'k', k);
    tmptag = zeros(nTest, k);
    for i = 1:k
        for j = 1:nTest
            tmptag(j, i) = fitrntag(idx(j, i));
        end
    end
    testtag(:, nClassier) = mode(tmptag,2);
    clear i j;
end

%% Post Step: Correct Rate & Analysis
% Calculate the mode of each test label result & correct rate
predtag = mode(testtag, 2);
nCorrect = 0;
for i = 1:nTest
    if(predtag(i) == tsttag(i))
        nCorrect = nCorrect + 1;
    end
end
accurcy = strcat(num2str(nCorrect/nTest*100),'%')
actural_dim = max(find(cumsum(latent)./sum(latent)<= 0.95))
