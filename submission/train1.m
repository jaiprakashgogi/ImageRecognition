function [ model ] = train1( X, Y )
    cell_size = 8;
    addpath('../vlfeat-0.9.20/toolbox/');
    vl_setup;
    
    num_imgs = size(X, 1);
    num_orientations = 21;
    feature_size = (32*32/(cell_size*cell_size)) * 4 * num_orientations;
    
    hogs = zeros(num_imgs, feature_size);
    orientation = zeros(num_imgs, 1);
    for i = 1:num_imgs
        if(mod(i, 100) == 0)
            disp('Done with 100 images');
        end
        img = single(reshape(X(i, :), 32, 32, 3));
        h = vl_hog(img, cell_size, 'variant', 'dalaltriggs', 'numOrientations', num_orientations);
        hogs(i, :) = reshape(h, 1, feature_size);
        
        %img_flip = flip(img, 1);
        %h = vl_hog(img_flip, cell_size, 'variant', 'dalaltriggs', 'numOrientations', num_orientations);
        %hogs(i*2, :) = reshape(h, 1, feature_size);
    end
    
    model = struct('hogs', hogs, 'labels', Y, 'cell_size', cell_size, 'num_orientations', num_orientations);
end
