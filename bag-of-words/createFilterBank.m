function [filterBank] = createFilterBank() 
    % Code to generate reasonable filter bank

    scales = [3 7 11];
    angles = [0 pi/6 pi/4 pi/3 pi/2 2*pi/3 3*pi/4 5*pi/6];
    lambda = 3;

    filterBank = cell(length(scales) * length(angles), 1);

    idx = 0;

    for angle = angles
        for scale = scales
            idx = idx + 1;
            filterBank{idx} = generate_gabor_filter(scale, angle, 3, 0, 1);
        end
    end
return;
