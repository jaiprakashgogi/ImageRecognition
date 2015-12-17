function [ Y ] = classify2( model, X )
    centroids = model.centroids;
    whitener = model.whitener;
    mean = model.mean;
    normalize_mean = model.normalize_mean;
    normalize_std = model.normalize_std;
    svm = model.svm;
    
    X = double(X);

    features = generate_descriptor(X, centroids, mean, whitener);
    features = bsxfun(@rdivide, bsxfun(@minus, features, normalize_mean), normalize_std);
    features = [features, ones(size(features, 1), 1)];

    % test and print result
    [~, Y] = max(features*svm, [], 2);
    Y = Y - 1;
end

