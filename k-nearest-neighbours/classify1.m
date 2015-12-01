function [ Y ] = classify( model, X )
    NUM_VOTES = 51;    
    images = model.images;
    labels = model.labels;
    
    num_images = size(X, 1);
    Y = zeros(num_images, 1);
    for i = 1:num_images
        if mod(i, 100) == 0
            disp(i);
        end
        distances = pdist2(double(X(i, :)), double(images));
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