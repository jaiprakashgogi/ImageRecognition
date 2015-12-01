function [img] = row_to_image(pixels)
    % We know that the images are 32x32 and 3 channel
    img = reshape(pixels, 32, 32, 3);
end