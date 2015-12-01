addpath('../vlfeat-0.9.20/toolbox/');
run vl_setup.m;
cellSize = 8 ;
%
X = []; Y = [];
for j = 1:1
    ws = sprintf('../data/small_data_batch_%d.mat', j);
    load(ws);
    for i = 1:size(data,1)
        im = reshape(single(data(i, :)), [32 32 ,3]);
        hog = vl_hog(im, cellSize, 'verbose') ;
        X = cat(1, X, hog(:)');
    end
    Y = cat(1, Y, labels);
end

size(X)
%%
C1 = 0.2:0.1:0.9;
for i = 1:length(C1)
    C = C1(i);
    [ Model ] = train( X, Y, C);
    ws = sprintf('Model_%d.mat',i);
    save(ws, 'Model');
end

%%
load('Model_8.mat', 'Model');
Acc = [];
for i = 1:5
ws = sprintf('../data/small_data_batch_%d.mat', i);
load(ws);
[Y] = classify(Model, data);

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