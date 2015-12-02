function [ Y ] = classify1( model, X )
    addpath('../vlfeat-0.9.20/toolbox/');
    vl_setup;

    NUM_VOTES = 51;    
    hogs = model.hogs;
    labels = model.labels;
    cell_size = model.cell_size;
    
    num_images = size(X, 1);
    Y = zeros(num_images, 1);
    for i = 1:num_images
        if mod(i, 100) == 0
            disp(i);
        end
        h = vl_hog(single(reshape(X(i, :), 32, 32, 3)), cell_size);
        h = reshape(h, 1, 496);
        distances = pdist2(double(h), double(hogs));
        [smallest, idx] = getNElements(distances, NUM_VOTES);
        
        votes = labels(idx);
        Y(i) = mode(votes);
    end
end

function [smallestNElements smallestNIdx] = getNElements(A, n)
     [ASorted AIdx] = sort(A);
     smallestNElements = ASorted(1:n);
     smallestNIdx = AIdx(1:n);
end