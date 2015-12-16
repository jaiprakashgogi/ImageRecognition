addpath('../../vlfeat-0.9.20/toolbox/');
run vl_setup.m;
cellSize = 8 ;
%
X = []; Y = [];
for j = 1:5
    ws = sprintf('../../data/small_data_batch_%d.mat', j);
    load(ws);
    for i = 1:size(data,1)
        i
        x = vector2im(data(i,:));
        X = cat(1, X, x);
    end
    Y = cat(1, Y, labels);
end
%%
C = 0.1;
[ Model ] = train_svm_batch( X, Y, C);
save('Model.mat', 'Model');
%%
%load('../data/small_data_batch_1.mat');
load('Model.mat');  
%classify_svm_batch(Model, data)

Acc = [];
for j = 1:5
ws = sprintf('../data/small_data_batch_%d.mat', j);
load(ws);
[Y] = classify_svm_batch(Model, data);

% confusion matrix
uniq =  unique(labels);
C = zeros(length(uniq));
for i = 1:size(Y,1)
    C(Y(i) + 1, labels(i) +1) = C(Y(i) +1, labels(i)+1) + 1;
end

acc = trace(C)/sum(sum(C))
Acc = cat(1, Acc, acc);
end
Acc