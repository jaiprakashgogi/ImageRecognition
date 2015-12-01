function [ h ] = getImageFeatures( wordMap, dictSize )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    h = zeros(dictSize, 1);
    
    wm = reshape(wordMap, size(wordMap, 1) * size(wordMap, 2), 1);
    h = hist(wm, dictSize);
    
    % Use the L1 norm here
    h = (h') / sum(h);
end

