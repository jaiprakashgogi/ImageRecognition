training_set = [1, 2, 3, 4];

combined_data = zeros(4*1000, 3072);
combined_label = zeros(4*1000, 1);
for i = training_set
    file = ['../data/small_data_batch_', num2str(i), '.mat'];
    load(file);
    combined_data((i-1)*1000 + 1:i*1000, :) = data(:, :);
    combined_label((i-1)*1000+1:i*1000, :) = labels(:, :);
end
Model1 = train1(combined_data, combined_label);
save('./Model1.mat', 'Model1');

%%
test_set = 5;
load(['../data/small_data_batch_', num2str(test_set),'.mat']);
load('./Model1.mat');
guessed_Y = classify1(Model1, data);


% Check the accuracy
disp('Calculating accuracy');
num_images = size(data, 1);
count = 0;
for i = 1:num_images
    if labels(i) == guessed_Y(i)
        count = count + 1;
    end
end

disp(['Accuracy is ', num2str(count*100/num_images), '%']);