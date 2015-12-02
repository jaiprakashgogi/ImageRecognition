training_set = [1, 2, 3, 4, 5];

combined_data = zeros(4*1000, 3072);
combined_label = zeros(4*1000, 1);
for i = training_set
    file = ['../data/small_data_batch_', num2str(i), '.mat'];
    load(file);
    combined_data((i-1)*1000 + 1:i*1000, :) = data(:, :);
    combined_label((i-1)*1000+1:i*1000, :) = labels(:, :);
end

non_3 = (combined_label~=3);
non_4 = (combined_label~=4);
non_5 = (combined_label~=5);

Model1 = train1(combined_data(:, :), combined_label(:));
save('./Model1.mat', 'Model1');
disp('Done training');

%%
test_set = 5;
load(['../data/small_data_batch_', num2str(test_set),'.mat']);
load('./Model1.mat');
num_images = size(data, 1);
%num_images = 10;
guessed_Y = classify1(Model1, data(1:num_images, :));

% Check the accuracy
disp('Calculating accuracy');

confusion = zeros(10, 10);

count = 0;
for i = 1:num_images
    actual = labels(i)+1;
    predicted = guessed_Y(i)+1;
    
    confusion(actual, predicted) = confusion(actual, predicted) + 1;
    
    
    %if labels(i) == guessed_Y(i)
    %    count = count + 1;
    %else
    %    disp(['Invalid prediction ', num2str(i)]);
    %    labels(i)
    %    guessed_Y(i);
    %end
end

imagesc(confusion);

%disp(['Accuracy is ', num2str(count*100/num_images), '%']);
trace(confusion);

%%%% Things tried
% Gaussian sigma
