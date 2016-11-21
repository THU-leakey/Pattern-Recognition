function [bestacc,bestc,bestg] = FindBestcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%Find best c/g parameters for LibSVM in Guass Kernel

%% 
%
% THU2016 Foundations of Machine Learning assignment4 programming.
% Copyright Leakey, 20161119-20161122.
%
% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm
 
%% about the parameters of FindBestcg
if nargin < 10 % the number of input variables
    accstep = 4.5; % default
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end

%% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
eps = 10^(-4);
 
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        cg(i,j) = svmtrain(train_label, train, cmd);
         
        if cg(i,j) <= 55
            continue;
        end
         
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
         
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
         
    end
end

%% to draw the test_error with different c & g
figure;
subplot(2, 1, 1);
cg = 100 - cg;
error = 100 - bestacc;
[C,h] = contour(X,Y,cg,0:accstep:30);
clabel(C,h,'Color','r');
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
firstline = 'SVCErrorwithc/g(ContourView)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVError=',num2str(error),'%'];
title({
firstline;secondline
},'Fontsize',12);
grid on; 

subplot(2, 1, 2);
meshc(X,Y,cg);
axis([cmin,cmax,gmin,gmax,0,50]); % set x-y-z axis
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('Error(%)','FontSize',12);
firstline = 'SVCErrorwithc/g(3DView)[GridSearchMethod]'; 
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVError=',num2str(100 - bestacc),'%'];
title({
firstline;secondline
},'Fontsize',12);
