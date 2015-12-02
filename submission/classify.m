function [Y] = classify(Model, data)
cellSize = 8 ;
X = [];
for i = 1:size(data,1)
	im = reshape(single(data(i, :)), [32 32 ,3]);
	hog = vl_hog(im, cellSize, 'verbose') ;
	X = cat(1, X, hog(:)');
end

X = double(X);
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

