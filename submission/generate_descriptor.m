function descriptors = generate_descriptor(X, centroids, patch_mean, P)
    num_centroids = size(centroids, 1);
    num_images = size(X, 1);
  
    % compute features for all training images
    descriptors = zeros(num_images, num_centroids*4);
    for i=1:num_images
        if (mod(i,100) == 0)
            fprintf('Extracting features: %d / %d\n', i, num_images);
            %fflush(stdout);
        end
    
        img = reshape(X(i, :), 32, 32, 3);
        patches = [ im2col(squeeze(img(:, :, 1))); im2col(squeeze(img(:, :, 2))); im2col(squeeze(img(:, :, 3)))]';
    
        % Normalize
        epsilon = 10.0;
        patches_mean = mean(patches, 2);
        patches_var  = sqrt(var(patches, [], 2) + epsilon);
        patches = bsxfun(@minus, patches, patches_mean);
        patches = bsxfun(@rdivide, patches, patches_var);
        
        % Mahalanobis transform
        patches = bsxfun(@minus, patches, patch_mean) * P;
    
        % k-means triangle
        x_2 = sum(patches .^ 2, 2);
        c_2 = sum(centroids .^ 2, 2)';
        x_c = patches * centroids';
    
        % Proximity to the cluster
        z = sqrt(bsxfun(@plus, c_2, bsxfun(@minus, x_2, 2 * x_c)));
        mu = mean(z, 2);
        patches = max(bsxfun(@minus, mu, z), 0);
    
        patches = reshape(patches, 27, 27, num_centroids);
    
        % Pooling into a 2x2 grid
        top_left     = sum(sum(patches(1:14,   1:14, :),     1), 2);
        bottom_left  = sum(sum(patches(15:end, 1:14, :),     1), 2);
        top_right    = sum(sum(patches(1:14,   15:end, :),   1), 2);
        bottom_right = sum(sum(patches(15:end, 14+1:end, :), 1), 2);
    
        % Final feature vector!
        descriptors(i,:) = [top_left(:); bottom_left(:); top_right(:); bottom_right(:)]';
    end
end

