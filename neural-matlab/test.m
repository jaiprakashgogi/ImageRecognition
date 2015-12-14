addpath('../vlfeat-0.9.20/toolbox/');
run vl_setup.m;
patch = [];
Data = [];
for j = 1:5
    ws = sprintf('../data/small_data_batch_%d.mat', j);
    load(ws);
    for i = 1:size(data,1)
        tic
        i
        pat = extract_patch(data(i,:));
        toc
        patch = cat(1,patch, pat);
        %X = cat(1, X, x);
        Data = cat(1, Data, data(i,:));
    end
    %[Xwh, mu, invMat, whMat] = whiten(patch);
    % visualize_patch(patch, 6)
    % visualize_patch(data(i,:),patch(1+i*27*27:(i+1)*27*27,:), 6)
end

save('patch_1_5.mat', 'patch', 'Data');

patch(find(isinf(patch))) = 0;
patch(find(isnan(patch))) = 0;
[Xwh, mu, invMat, whMat] = whiten(patch);

% visualize_patch(data(i,:),patch(1+i*27*27:(i+1)*27*27,:), 6)
% visualize_patch(data(i,:),Xwh(1+i*27*27:(i+1)*27*27,:), 6)