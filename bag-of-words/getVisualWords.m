function [ wordmap ] = getVisualWords( I, filterBank, dictionary )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    %wordmap = zeros(size(I, 1), size(I, 2));

    responses = extractFilterResponses(I, filterBank);
    pixelCount = size(I,1)*size(I,2);
    num_filters = length(filterBank);
    
    img_height = size(I, 1);
    img_width = size(I, 2);
    
    dist = pdist2(responses, dictionary, 'euclidean');
    [~, wordmap] = min(dist');
    wordmap = reshape(wordmap', img_height, img_width);
end