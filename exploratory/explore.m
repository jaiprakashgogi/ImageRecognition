training_set = [1, 2, 3, 4, 5];

combined_data = zeros(4*1000, 3072);
combined_label = zeros(4*1000, 1);
for i = training_set
    file = ['../data/small_data_batch_', num2str(i), '.mat'];
    load(file);
    combined_data((i-1)*1000 + 1:i*1000, :) = data(:, :);
    combined_label((i-1)*1000+1:i*1000, :) = labels(:, :);
end

%%
% Compute the histograms
histogram(combined_label, 10);

% Calculate the average images
avg = cell(10);
for i = 0:9
    imgs = combined_data(combined_label==i, :);
    count = size(imgs, 1);
    avg_img = sum(imgs) / count;
    avg{i+1} = avg_img;
end

combo = zeros(64, 160, 3);
for i = 1:10
    starty = 1;
    startx = 32*(i-1) + 1;
    if i > 5
        starty = 33;
        startx = startx - 160;
    end
    
    combo(starty:starty+31, startx:startx+31, :) = imrotate(reshape(avg{i}, 32, 32, 3)/255, 270);
end
imshow(combo);

histogram_overall = zeros(3, 256);

hist_class = cell(10, 1);
for i = 1:10
    hist_class{i} = zeros(3, 256);
end

for i = 1:5000
    [hr, hg, hb] = img_hist(combined_data(i, :));
    
    hist_class{combined_label(i)+1} = hist_class{combined_label(i)+1} + [hr; hg; hb];
    histogram_overall = histogram_overall + [hr; hg; hb];
end
render_hist(histogram_overall);