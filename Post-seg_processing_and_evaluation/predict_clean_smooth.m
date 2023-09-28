%%
%main_histo_data = ['C:/Users/rxiao/Desktop/desktop/paper3/seg_result_4subtype/test'];
main_histo_data = ['D:/TCGA/seg_result/rep9/RCC_TCGA/SSL_multi'];

list_images = dir([main_histo_data '/*.png']);
n_images = length(list_images);

for idx_image = 1:n_images    
        fprintf('Processing image %d/%d... ', idx_image, n_images);
        img_name = list_images(idx_image).name;
        image_path = [main_histo_data '/' img_name];
        img = rescale01(double(imread(image_path)));
        clean_path = [main_histo_data '/predict_clean/' img_name];
        clean_skel_path=[main_histo_data '/predict_clean_skel/' img_name];
        clean_skel_prune_path = [main_histo_data '/predict_clean_prune_skel/' img_name];
        smooth_path = [main_histo_data '/predict_smooth/' img_name];
        
        LabelImage=imread(image_path);
        %LabelImage_gray = rgb2gray(LabelImage);
        LabelImage_gray = LabelImage;
        LabelImage_gray(LabelImage_gray<255) = 0;
        LabelImage_gray(LabelImage_gray==255) = 1;
        cc=bwconncomp(LabelImage_gray);  %find target grid 
        matrix_skel=labelmatrix(cc);   %assign target label 
        L = bwlabeln(matrix_skel, 4);
        S = regionprops(L, 'Area');
%        bw1 = ismember(L, find([S.Area] >= 500));
        bw1 = ismember(L, find([S.Area] >= 300));
        imwrite(mat2gray(bw1), clean_path);
        
        i=imread(clean_path);

        i1=i;
        %imshow(i1)
%        se=strel("disk",2);
        se=strel("disk",1);
        i2=im2bw(i1);    %binary 
        i3 = imclose(i2,se);  %close operation
        i4 = imopen(i3,se);  %open operation
        imwrite(mat2gray(i4), smooth_path);
        

        img_mask = imread(smooth_path);
    %    I = rgb2gray(img_mask);
        I = img_mask;
        bw1=imbinarize(I,0.1);
        se1=strel('disk',2);
    %      bw1=imbinarize(I,0.01);
    %      se1=strel('disk',5);
        A1=imerode(bw1,se1);
        out = bwskel(A1);
       out = im2uint8(out);
       imwrite(out,clean_skel_path);        
        
               
        clean_skel=imread(clean_skel_path);
        out = bwskel(logical(clean_skel),'MinBranchLength',10);%modified skel
        imwrite(out,clean_skel_prune_path);
        fprintf('Done!\n');
        
end