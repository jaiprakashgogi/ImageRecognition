addpath('../svm');
%patch = [];
X = []; Y = [];
for j = 1:5
    ws = sprintf('../data/small_data_batch_%d.mat', j);
    load(ws);
    for i = 1:size(data,1)
        i
        pat = preporcess_normalize_whiten(data(i,:));
        X = cat(1,X,pat);
    end
    Y = cat(1, Y, labels);
end
%%
C = 0.1;
[ Model ] = train_svm_batch( X, Y, C);
save('Model3.mat', 'Model');
save('XY.mat', 'X', 'Y');
%%
clear all;
close all;
clc;
load('Model3.mat');  

Acc = [];
for i = 1:5
ws = sprintf('../data/small_data_batch_%d.mat', i);
load(ws);
[Y] = classify3(Model, data);

% confusion matrix
uniq = unique(labels);
C = zeros(length(uniq));
for i = 1:size(Y,1)
    C(Y(i) + 1, labels(i) +1) = C(Y(i) +1, labels(i)+1) + 1;
end

acc = trace(C)/sum(sum(C))
Acc = cat(1, Acc, acc);
end
Acc