function [ hr, hg, hb ] = img_hist( img )
    img_sq = reshape(img, 32, 32, 3);
    c1 = squeeze(img_sq(:, :, 1));
    c2 = squeeze(img_sq(:, :, 2));
    c3 = squeeze(img_sq(:, :, 3));
    
    hr = zeros(1, 256);
    hg = zeros(1, 256);
    hb = zeros(1, 256);
    
    for i = 0:255
        hr(i+1) = sum(sum(c1==i));
        hg(i+1) = sum(sum(c2==i));
        hb(i+1) = sum(sum(c3==i));
    end
end