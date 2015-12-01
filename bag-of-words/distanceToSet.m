function [ histInter ] = distanceToSet( wordHist, histograms )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    % wordHist = col vector
    % histograms = each training image is in a row
    minimums = bsxfun(@min, wordHist', histograms);
    output = min(minimums', histograms')';

    histInter = sum(output, 2);
end

