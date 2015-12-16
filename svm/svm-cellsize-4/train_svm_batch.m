function [ Model ] = train_svm_batch( X, Y ,C)
labels = unique(Y);
for i = 1:length(labels)
    count = 1;
    for j = 1:length(labels)
        if i ~= j
            idx_pos = find(Y==labels(i));
            idx_neg = find(Y==labels(j));
            x_pos = X(idx_pos,:);
            x_neg = X(idx_neg,:);
            label_pos = ones(length(idx_pos),1);
            label_neg = -1*ones(length(idx_neg),1);
            x = [x_pos; x_neg];
            label = [label_pos; label_neg];
            [ w, b ] = svm_solver(x,label,C);
            model.w = w;
            model.b = b;
            model.labels = labels;
            class{count} = model; 
            count = count + 1;
        end
    end
    Model{i} = class;
end
end