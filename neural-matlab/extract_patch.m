function patch = extract_patch(data)
w = 6;
s = 1; %stride
im = reshape(data, [32 32 ,3]);
im = double(im);
patch = [];
for y = 1:s:size(im,1)-w+1
	for x = 1:s:size(im,2)-w+1
        pat = im(y:y+w-1, x:x+w-1, :);
        mean_pat = pat(:) - mean(pat(:));
        std_pat = sum(mean_pat.^2)/length(pat(:));
        patch = cat(1, patch, (mean_pat/(sqrt(std_pat + eps)))');
	end
end
    
end

