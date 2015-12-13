load('../data/small_data_batch_1.mat');
fp = fopen('../data/small_data_batch_1.bin', 'w');

for i = 1:1000
    fwrite(fp, labels(i), 'uchar');
    fwrite(fp, data(i, :), 'uchar');
end

fclose(fp);