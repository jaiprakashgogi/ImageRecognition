function [ Y ] = classify1( model, X )
    addpath('../vlfeat-0.9.20/toolbox/');
    vl_setup;

    NUM_VOTES = 5;    
    hogs = model.hogs;
    labels = model.labels;
    cell_size = model.cell_size;
    num_orientations = model.num_orientations;
    
    feature_size = (32*32)/(cell_size*cell_size) * 4 * num_orientations;
    
    num_images = size(X, 1);
    Y = zeros(num_images, 1);
    for i = 1:num_images
        if mod(i, 100) == 0
            disp(i);
        end
        
        img = single(reshape(X(i, :), 32, 32, 3));
        
        h = vl_hog(img, cell_size, 'variant', 'dalaltriggs', 'numOrientations', num_orientations);
        h = reshape(h, 1, feature_size);
        distances = pdist2(double(h), double(hogs));
        [~, idx] = getNElements(distances, NUM_VOTES);
        
        votes = labels(idx);
        Y(i) = mode(votes);
    end
end

function [smallestNElements smallestNIdx] = getNElements(A, n)
     [ASorted AIdx] = sort(A);
     smallestNElements = ASorted(1:n);
     smallestNIdx = AIdx(1:n);
end