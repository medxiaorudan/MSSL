clear all;
close all;
clc;

% Ignoring the 'image is too big to fit on screen' warnings not to be
% flooded.
warning off images:initSize:adjustingMag

main_histo_data = ['D:/TCGA/use/order_RCC_TCGA'];
%dir_names = {'seg'};
case_folder = [main_histo_data, '/seg/'];
image_list = dir([case_folder, '/*.png']);
n_images = length(image_list);
for idx_image = 1:n_images
    fprintf('Processing image %d/%d... ', idx_image, n_images);
    mask_image_name = image_list(idx_image).name;
    image_mask_path = [case_folder, mask_image_name];
    img_mask = imread(image_mask_path);
%    I = rgb2gray(img_mask);
    I = img_mask;
    bw1=imbinarize(I,0.1);
    se1=strel('disk',2);
    A1=imerode(bw1,se1);
    out = bwskel(A1);
   out = im2uint8(out);
    imwrite(out, [main_histo_data,'/skel/' mask_image_name]);
    fprintf('Done!\n');
end

