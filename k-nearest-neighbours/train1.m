function [ model ] = train1( X, Y )
    cell_size = 8;
    addpath('../vlfeat-0.9.20/toolbox/');
    vl_setup;
    
    tic;
    num_imgs = size(X, 1);
    hogs = zeros(num_imgs, 496);
    for i = 1:num_imgs
        h = vl_hog(single(reshape(X(i, :), 32, 32, 3)), cell_size);
        hogs(i, :) = reshape(h, 1, 496);
    end
    toc;
    
    model = struct('hogs', hogs, 'labels', Y, 'cell_size', cell_size);
end