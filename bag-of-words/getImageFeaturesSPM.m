function [ h ] = getImageFeaturesSPM( layerNum, wordMap, dictSize )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    % Evaluate the finest level first
    img_height = size(wordMap, 1);
    img_width = size(wordMap, 2);
    
    h = zeros(dictSize * ((4^layerNum) - 1) / 3, 1);
    
    weights = [0.25, 0.25, 0.5];
    
    for layer = layerNum:-1:1
        max_split = 2^(layer-1);
        starts_at = floor(dictSize * ((4^(layer-1)) - 1) / 3);

        cell_width = floor(img_width / max_split);
        cell_height = floor(img_height / max_split);

        for row = 1:max_split
            for col = 1:max_split
                subregion = wordMap(cell_height*(row-1)+1:cell_height*row, cell_width*(col-1)+1:cell_width*col);
                
                % The beginning/ending in the feature vector
                starting = floor(starts_at + max_split*dictSize*(col-1) + dictSize*(row-1) + 1);
                ending = floor(starts_at + max_split*dictSize*(col-1) + dictSize*(row));
                
                h( starting:ending, 1) = getImageFeatures(subregion, dictSize) * weights(layer);
            end
        end
    end
    
    h = h / sum(h);
end

