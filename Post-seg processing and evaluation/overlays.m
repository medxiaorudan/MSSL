clear all;
close all;
clc;

% Ignoring the 'image is too big to fit on screen' warnings not to be
% flooded.
warning off images:initSize:adjustingMag

% Get the current file path and add the external
% functions to the matlab path so they can be used in this script.
[pathstr, ~, ~] = fileparts(mfilename('D:/test_vascular_mask/overlays.m'));
addpath(genpath(pathstr));

%% Loading the test image
main_histo_data = [pathstr 'D:/test_vascular_mask/ccRCC'];
dir_names = {'16','17','18'};
n_dir = numel(dir_names);
%mask = 'mask';
tic;

for i_dir = 1:n_dir
    case_folder = [main_histo_data, '/', dir_names{i_dir}];
    list_images = dir([case_folder '/subimages/*.png']);
    n_images = length(list_images);
    
    for idx_image = 1:n_images    
        fprintf('Processing image %d/%d... ', idx_image, n_images);
        img_name = list_images(idx_image).name;
        image_path = [case_folder '/subimages/' img_name];
        img = rescale01(double(imread(image_path)));
        skeleton_path = [case_folder, '/skeletons/skeleton_segmentation_' img_name];
        if exist(skeleton_path, 'file')
%           figure, imshow(img);
           skeleton = imread(skeleton_path);
%            dilated_skel = imdilate(skeleton, ones(5));
%            ov = overlay(img, dilated_skel, [0 1 0]);
        end
        dilated_skel = imdilate(skeleton, ones(5));
        ov = overlay(img, dilated_skel, [0 1 0]);
        imwrite(ov, [case_folder '/overlay/overlay_' img_name]);
        fprintf('Done!\n');
    end
end