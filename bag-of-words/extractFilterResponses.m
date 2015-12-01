function [filterResponses] = extractFilterResponses(I, filterBank)
% CV Fall 2015 - Provided Code
% Extract the filter responses given the image and filter bank
% Pleae make sure the output format is unchanged. 
% Inputs: 
%   I:                  a 3-channel RGB image with width W and height H 
%   filterBank:         a cell array of N filters
% Outputs:
%   filterResponses:    a W*H x N*3 matrix of filter responses
 

    %Convert input Image to Lab
    doubleI = double(I);
    pixelCount = size(doubleI,1)*size(doubleI,2);

    if ndims(I) == 3
        [L,a,b] = RGB2Lab(doubleI(:,:,1), doubleI(:,:,2), doubleI(:,:,3));
        num_channels = 3;
    else
        L = I;
        num_channels = 1;
    end

    %filterResponses:    a W*H x N*3 matrix of filter responses
    filterResponses = zeros(pixelCount, length(filterBank)*num_channels);

    %for each filter and channel, apply the filter, and vectorize

    % === fill in your implementation here  ===
    % Number of filters
    num_filters = length(filterBank);

    for f = 1:num_filters
        for c = 1:num_channels
            if c==1
                out = imfilter(L, filterBank{f}, 'symmetric');
            end
            if c==2
                out = imfilter(a, filterBank{f}, 'symmetric');
            end
            if c==3
                out = imfilter(b, filterBank{f}, 'symmetric');
            end
        
            out = reshape(out, pixelCount, 1);
            filterResponses(:, num_channels*(f-1) + c) = out;
        end
    end
    
    if num_channels == 1
        filterResponses = repelem(filterResponses, 1, 3);
    end
end