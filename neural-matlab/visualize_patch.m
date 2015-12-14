function visualize_patch(patch, w)
img = [];
    X = [];
    for i = 1:size(patch,1)
        X = cat(2, X, reshape(patch(i,:), [w w 3]));
        if rem(i,27) ==0
            img = cat(1, img, X);
            X = [];
        end
    end
imagesc(img);
end
