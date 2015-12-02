function [ model ] = train( X, Y )
    % Constants that can be tweaked
    NUM_CLUSTERS = 150;
    num_images = size(X, 1);
    
    filter_bank = createFilterBank();
    responses = zeros(1024 * num_images, 72);
    disp('Calculating filter responses');
    tic
    for i = 1:num_images
        if mod(i, 100) == 0
            disp(i);
        end
        responses((i-1)*1024+1:i*1024, :) = extractFilterResponses(reshape(X(i, :), 32, 32, 3), filter_bank);
    end
    toc
    
    disp(['Starting K-Means with ', num2str(NUM_CLUSTERS), ' clusters']);
    tic
    [~, words] = kmeans(responses, NUM_CLUSTERS);
    toc
    
    disp('Evaluating features');
    tic
    num_words = size(words, 1);
    layers = 3;
    feature_size = num_words * ( (4^(layers) - 1) / 3 );
    train_features = zeros(num_images, feature_size);
    for i = 1:num_images
        if mod(i, 100) == 0
            disp(i);
        end
        wm = getVisualWords( reshape(X(i, :), 32, 32, 3), filter_bank, words );
        train_features(i, :) = getImageFeaturesSPM(layers, wm, num_words);
    end
    toc
    
    model = struct('dictionary', words, 'features', train_features, 'labels', Y, 'k', NUM_CLUSTERS);
end