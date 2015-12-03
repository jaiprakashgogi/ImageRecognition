function [Y] = classify(Model, data)
X = [];
for i = 1:size(data,1)
	x = vector2im(data(i,:));
	X = cat(1, X, x);
end

nClass = length(Model);
nSubClass = length(Model{1});

X = double(X);
y = zeros(size(X,1), nClass, nSubClass);
for n = 1:size(X,1)
    for i = 1:nClass
        for j = 1:nSubClass
            model = Model{i}{j};
            y(n,i,j) = model.w*X(n,:)' + model.b;
        end
    end
end


label = zeros(size(X,1), nClass);
for n = 1:size(X,1)
    for i = 1:nClass
        p = y(n,i,:);
        p = p(:);
        label(n,i) = length(find(p>0));
    end
end

[~,idx] = max(label');
Y = idx' - ones(size(idx',1), 1);

end
