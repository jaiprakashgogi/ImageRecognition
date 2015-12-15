function out = preporcess_normalize_whiten(data)
load('jp_white.mat');
load('mat.centroids.mat');
patch = extract_patch(data);
pat = patch*whitener;
dist = pdist2(pat,centroids);
[~, out] = min(dist');    
end

