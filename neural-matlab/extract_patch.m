function patch = extract_patch(data)
w = 6;
s = 1; %stride
im = reshape(data, [32 32 ,3]);
im = im2double(im);
patch = [];
for c = 1:size(im,3)
    channel = [];
    for y = 1:s:size(im,1)-w+1
        for x = 1:s:size(im,2)-w+1
            pat = im(y:y+w-1, x:x+w-1, c);
            channel = cat(1, channel, ((pat(:) - mean(pat(:)))/(std(pat(:)+eps)))');
        end
    end
    patch = cat(2, patch, channel);
end
end

