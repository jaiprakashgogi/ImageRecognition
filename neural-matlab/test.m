addpath('../vlfeat-0.9.20/toolbox/');
run vl_setup.m;
patch = [];
for j = 1:1
    ws = sprintf('../data/small_data_batch_%d.mat', j);
    load(ws);
    for i = 1:size(data,1)
        tic
        i
        pat = extract_patch(data(i,:));
        toc
        patch = cat(1,patch, pat);
        %X = cat(1, X, x);
    end
    %[Xwh, mu, invMat, whMat] = whiten(patch);
    %visualize_patch(patch, 6)
end
