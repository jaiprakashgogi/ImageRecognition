load('../data/small_data_batch_1.mat');
fp = fopen('../data/small_data_batch_1.bin', 'w');

for i = 1:1000
    fwrite(fp, labels(i), 'uchar');
    fwrite(fp, data(i, :), 'uchar');
end

fclose(fp);

load('../data/small_data_batch_2.mat');
fp = fopen('../data/small_data_batch_2.bin', 'w');

for i = 1:1000
    fwrite(fp, labels(i), 'uchar');
    fwrite(fp, data(i, :), 'uchar');
end

fclose(fp);

load('../data/small_data_batch_3.mat');
fp = fopen('../data/small_data_batch_3.bin', 'w');

for i = 1:1000
    fwrite(fp, labels(i), 'uchar');
    fwrite(fp, data(i, :), 'uchar');
end

fclose(fp);

load('../data/small_data_batch_4.mat');
fp = fopen('../data/small_data_batch_4.bin', 'w');

for i = 1:1000
    fwrite(fp, labels(i), 'uchar');
    fwrite(fp, data(i, :), 'uchar');
end

fclose(fp);

load('../data/small_data_batch_5.mat');
fp = fopen('../data/small_data_batch_5.bin', 'w');

for i = 1:1000
    fwrite(fp, labels(i), 'uchar');
    fwrite(fp, data(i, :), 'uchar');
end

fclose(fp);