clear all;
close all;
clc;

% Ignoring the 'image is too big to fit on screen' warnings not to be
% flooded.
warning off images:initSize:adjustingMag

% Get the current file path and add the external
% functions to the matlab path so they can be used in this script.
[pathstr, name, ext] = fileparts(mfilename('D:/test_vascular_mask/mask_skeleton.m'));
addpath(genpath(pathstr));
%%
main_histo_data = [pathstr 'D:/test_vascular_mask/pRCC'];
dir_names = {'0','1','6','7','8'};
n_dir = numel(dir_names);

%%
for i_dir = 1:n_dir
    case_folder = [main_histo_data, '/', dir_names{i_dir},'/','subimages'];
    image_list = dir([case_folder,'/', '*.png']);
    n_images = length(image_list);
    fprintf('total image: %d', n_images);
    for idx_image = 1 : n_images
        fprintf('Processing image %d/%d... ', idx_image, n_images);
        image_name = image_list(idx_image).name;
        image_path = [case_folder,'/', image_name];
        img = imread(image_path);
        [V,H,C] = deconvolve(img);
        [N,visu] = DetectNuclei(H,C);    
        imwrite(visu, [main_histo_data, '/', dir_names{i_dir},'/nuclei/nuclei_', image_name]);
        fprintf('Done!\n');
    end
end
