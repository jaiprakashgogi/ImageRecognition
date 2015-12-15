%%


%%
% At this point, the individual rows are stored in row-major form. These
% need to convert to the column major form used by Matlab.

for y = 1:num_rows
    out = zeros(6, 6, 3);
    
    chB = reshape(centroids(y, 1:3:108), 6, 6);
    chG = reshape(centroids(y, 2:3:108), 6, 6);
    chR = reshape(centroids(y, 3:3:108), 6, 6);
    
    %out(:, :, 1) = reshape(img(3:3:108), 6, 6);
    %out(:, :, 2) = reshape(img(2:3:108), 6, 6);
    %out(:, :, 3) = reshape(img(1:3:108), 6, 6);
    
    out(:, :, 1) = chR';
    out(:, :, 2) = chG';
    out(:, :, 3) = chB';
    centroids(y, :) = reshape(out, 1, 108);
end

save('./mat.centroids.mat', 'centroids');

%%
imshow(im2uint8(reshape(centroids(4, :), 6, 6, 3) * 255 + 128));
