I = imread('lena.jpg');
I = im2double(I);
imshow(I)

I_mean = I;
I_stf = I;
for i = 1:size(I,3)
    temp = I(:,:,i);
    temp1 = I(:,:,i) - mean(temp(:));
    I_mean(:,:,i) = temp1;
    std = sqrt(sum(temp1(:).^2)/length(temp1(:)) );
    I_std(:,:,i) = temp1/std;
end

figure, imshow(I_mean);
figure, imshow(I_std);
imwrite(I_mean,'lena_mean.jpg')
imwrite(I_std,'lena_std.jpg')