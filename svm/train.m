function [ Model ] = train( X, Y )
labels = unique(Y);
C = 0.1;
for i = 1:length(labels)
    label = double(Y);
    label(find(Y==labels(i))) = 1;
    label(find(Y~=labels(i))) = -1;
	[ w, b ] = svm_solver(X,label,C);
    model.w = w;
    model.b = b;
    model.labels = labels;
    Model{i} = model;
end
end

