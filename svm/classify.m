function [Y] = classify(Model, X)

y = zeros(size(X,1), length(Model));
for n = 1:size(X,1)
    for i = 1:length(Model)
        model = Model{i};
        y(n,i) = model.w*X(n,:)' + model.b;
    end
end

[~,idx] = max(y');
Y = idx' - ones(size(idx',1), 1);

end

