addpath('../vlfeat-0.9.20/toolbox/');
run vl_setup.m;
cellSize = 8 ;
%
X = []; Y = [];
for j = 1:5
    ws = sprintf('../data/small_data_batch_%d.mat', j);
    load(ws);
    for i = 1:size(data,1)
        im = reshape(single(data(i, :)), [32 32 ,3]);
        hog = vl_hog(im, cellSize, 'verbose') ;
        X = cat(1, X, hog(:)');
        % imhog = vl_hog('render', hog, 'verbose') ;
        % clf ; imagesc(imhog) ; colormap gray ;
    end
    Y = cat(1, Y, labels);
end

size(X)
%%
[ Model ] = train( X, Y );
save('Model.mat', 'Model');

%%
load('Model.mat');
[Y] = classify(Model, X);

%% confusion matrix
for i = 1:size(X,1)
    
end