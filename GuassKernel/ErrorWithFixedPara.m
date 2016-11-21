%% Q4
% 1. Set a fixed parametre
% 2. Get the training and testing errors as a function of training set
% 3. Plot the training error
% 4. Plot the test error
% 5. Estimate the asymptotic test error

%% Step 1
% first run PlotTestError.m
% set the fixed parametre
% choose guassian kernel: gamma = 0.0001, c = 1.678
% accuracy 1 ~ 5 are the testing accuracy.

% use 500/1000/1500/2000/2500/3000/3500/4000/4500 as training set
% use last 500 as testing set

%% Step 2
% get the training accuracy use model 1 - 5
test_error = [];
train_error = [];
test_inst = data5k_inst(4501:5000, :);
test_label = data5k_label(4501:5000, :);

for Ntrain = 500:500:4500
    train_inst = data5k_inst(1:Ntrain, :);
    train_label = data5k_label(1:Ntrain, :);
    model = svmtrain(train_label, train_inst, '-t 2 -g 0.0001 -b 1 -c 1.678'); % train and test model 
    [~, train_accuracy, ~] = svmpredict(train_label, train_inst, model, '-b 1');
    train_error = [train_error, 1 - train_accuracy(1)/100];
    [~, test_accuracy, ~] = svmpredict(test_label, test_inst, model, '-b 1');
    test_error = [test_error, 1 - test_accuracy(1)/100];
end
Ntrain = 500:500:4500;
figure();
plot(Ntrain, train_error);
figure();
plot(Ntrain, test_error);
