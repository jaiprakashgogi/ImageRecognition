function [ Y ] = classify( model, X )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    NUM_CLUSTERS = model.k;
    num_images = size(X, 1);
    
    Y = zeros(num_images, 1);
    dict_size = size(model.dictionary, 1);

    filter_bank = createFilterBank();
    for i = 1:num_images
        if(mod(i, 100) == 0)
            disp(i)
        end
        wm = getVisualWords( reshape(X(i, :), 32, 32, 3), filter_bank, model.dictionary );
        h = getImageFeaturesSPM(3, wm, dict_size);
        distances = distanceToSet(h, model.features);
        [~, nn] = max(distances);
        Y(i) = model.labels(nn);
    end
end

