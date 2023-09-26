% path_gt = "C:/Users/rxiao/Desktop/desktop/paper2/seg/seg_test_512 - 副本";
% path_test = 'C:/Users/rxiao/Desktop/desktop/paper2/seg_pred_supervise_ResNet/test/test';
% path_CRF = 'C:/Users/rxiao/Desktop/desktop/paper2/seg_pred_supervise_ResNet/test/CRF';

%%
% main_histo_data = ['C:/Users/rxiao/Desktop/desktop/paper2'];
% case_folder = [main_histo_data, '/seg_result_2/val/SSL_multi/'];
% GT_folder = [main_histo_data, '/seg/seg_val_512 - 副本/'];
% feature_path=[main_histo_data, '/evaluation_results/val/SSL_multi_rep2_4.txt'];
% fid=fopen(feature_path,'a+');
% fprintf(fid,'%s \t %s \t %s \t %s \t \n', 'ID','MR','FA','OurIndex'); 
% fclose(fid);
% 
% image_list = dir([case_folder, '/新建文件夹 (11)/*.png']);
% n_images = length(image_list);
% 
% s = struct;
% s.MR = 0;
% s.FA = 0;
% s.OurIndex = 0;

%%

GT = imread("C:/Users/rxiao/Desktop/desktop/paper3/Over_lay_result/gt_small.png");
I = imread("C:/Users/rxiao/Desktop/desktop/paper3/Over_lay_result/seg_small.png");
GT = GT>0;
I = I>0;

%figure;
Dif1 = double(GT)-double(I);
%imagesc(Dif1);

d3 = strel('disk',3);
ID = imdilate(I,d3);
d2 = strel('disk',2);

%M = (GT>ID);   
M = zeros(512,512);
for i =1:512
    for j=1:512
        if (ID(i,j)==0)&&(GT(i,j)~=0)
            M(i,j) = 1;
        end
    end
end
Missed = imopen(M,d2);
figure;
imagesc(GT);
gk = sum(GT(:));
figure;
imagesc(I);
ik = sum(I(:));
%%
GTD = imdilate(GT,d3);

E = zeros(512,512);



for i =1:512
    for j=1:512
        if (I(i,j)~=0)&&(GTD(i,j)==0)
            E(i,j) = 1;
        end
    end
end
Extra = imopen(E,d2);

% figure;
% imagesc(Extra);

mk = sum(Missed(:));
gk = sum(GT(:));
fk = sum(Extra(:));
ik = sum(I(:));

MR = sum(Missed(:))/sum(GT(:));
FA = sum(Extra(:))/sum(I(:));

OurIndex = 1- (MR+FA)/2;

TP=sum(bwskel(logical(ID)))-sum(bwskel(logical(Extra)));
FP=sum(bwskel(logical(Extra)));
FN=sum(bwskel(logical(Missed)));
f1_score=sum(TP)/(sum(TP)+0.5*(sum(FP)+sum(FN)));
%%
%plot beautiful images
Beauty = 255*ones(512,512,3);
temp = ones(512,512);
Beauty(:,:,2) = Beauty(:,:,2).*(temp-Extra);
Beauty(:,:,3) = Beauty(:,:,3).*(temp-Extra);
Beauty(:,:,1)  = Beauty(:,:,1).*(temp-Missed);
Beauty(:,:,2)  = Beauty(:,:,2).*(temp-Missed);
figure;
imagesc(Beauty);
% 
imwrite(Beauty,"C:/Users/rxiao/Desktop/desktop/paper2/seg/beautiful_plot/oncocytoma_8_overlay.png")
