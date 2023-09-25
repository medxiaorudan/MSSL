import os
import sys
import tarfile
import json
import cv2

import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image, ImageFile
from skimage.morphology import thin
from six.moves import urllib

from utils.mypath import MyPath, PROJECT_ROOT_DIR

import math

ImageFile.LOAD_TRUNCATED_IMAGES = True

def make_dataset(root, prefix=('jpg', 'png')):
    img_label_path = root[0]
    gt_label_path = root[1]
    img_list = [os.path.splitext(f)[0] for f in os.listdir(img_label_path) if f.endswith(prefix[1])]
    return [(os.path.join(img_label_path, img_name + prefix[1]), os.path.join(gt_label_path, img_name + prefix[1])) for img_name in img_list]

class RCC_context_classify(data.Dataset):

    VOC_CATEGORY_NAMES = [
                          'vascular']

    CONTEXT_CATEGORY_LABELS = [0,
                               1]

    def __init__(self,
                 root='/data/morpheme/user/rxiao/MIDL/bigdata/',
                 download=False,
                 split='val',
                 transform=None,
                 area_thres=0,
                 retname=True,
                 overfit=False,
                 do_SSL=False,
                 do_classify=False,
                 split_rate=(1, 3),
                 length=None
      
                 ):
        self.length=length
        self.split_rate = split_rate
        self.r_l_rate = split_rate[1] // split_rate[0]
        self.root = root

        self.transform = transform

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.area_thres = area_thres
        self.retname = retname


        self.classify = do_classify
        self.SSL = do_SSL
        self.train_labeled_path = [os.path.join(self.root, 'train', 'img_labeled'),os.path.join(self.root, 'train', 'seg')]

        self.train_unlabeled_path = [os.path.join(self.root, 'train', 'img_unlabeled'),os.path.join(self.root, 'train', 'img_unlabeled')]
        self.train_path = [os.path.join(self.root, 'train', 'img'),os.path.join(self.root, 'train', 'img')]
        self.val_path = [os.path.join(self.root, 'val', 'img_labeled'),os.path.join(self.root, 'val', 'seg')]
        self.test_path = [os.path.join(self.root, 'test', 'img'),os.path.join(self.root, 'test', 'seg')]
        print("Initializing dataloader for RCC {} set".format(''.join(self.split)))
        for splt in self.split:        
        
            if splt == "train":
                
                self.imgs_labeled = make_dataset(self.train_labeled_path, prefix=('.jpg', '.png'))
                
                len_labeled = len(self.imgs_labeled)    
        
                self.imgs_unlabeled = make_dataset(self.train_unlabeled_path, prefix=('.jpg', '.png'))
                
                len_unlabeled = self.r_l_rate * len_labeled
                
                self.length = len_labeled + len_unlabeled
                
                self.imgs_unlabeled = self.imgs_unlabeled * (self.r_l_rate + math.ceil(len_labeled / len_unlabeled))  # 扩展无标签的数据列表
                self.imgs_unlabeled = self.imgs_unlabeled[0:len_unlabeled]
    
            elif splt == "val":
            
                self.imgs_labeled = make_dataset(self.val_path, prefix=('.jpg', '.png'))
                len_labeled = len(self.imgs_labeled)

                self.length = len_labeled

            elif splt == "test":
            
                self.imgs_labeled = make_dataset(self.val_path, prefix=('.jpg', '.png'))
                len_labeled = len(self.imgs_labeled)
                self.length = len_labeled
            
            print('Number of dataset images: {:d}'.format(self.length))
            
    def __getitem__(self, index):
        sample = {}
        for splt in self.split: 
                        
            if splt == "train":
                if index % (self.r_l_rate + 1) == 0:
                    labeled_index = index // (self.r_l_rate + 1)
                    img_path, gt_path = self.imgs_labeled[labeled_index]  # 0, 1 => 10550
                    
                else:
                    unlabeled_index = (index //(self.r_l_rate + 1)) * self.r_l_rate + index % (self.r_l_rate + 1)-1
                    img_path, gt_path = self.imgs_unlabeled[unlabeled_index]
                
                img_name = (img_path.split("/")[-1]).split('.')[0]  
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1    
                         
                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)             
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32)//225
                sample =  {'image': img, "classify": label, "name": img_name, 'SSL_S':gt}

                if self.classify:
                    sample= {'image': img, "classify": label, "name": img_name}
                elif self.SSL:
                    sample = {'image': img, 'SSL': gt, "name": img_name}
                    sample["idx"] = index 

                if self.transform is not None:
                    sample= self.transform(sample)
                                          
            elif splt == "val":
                img_path, gt_path = self.imgs_labeled[index]  
                img_name = (img_path.split("/")[-1]).split('.')[0] 
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1
                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)             
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32)//225

                sample =  {'image': img, "classify": label, "name": img_name, 'SSL_S':gt}   

                if self.transform is not None:
                    sample= self.transform(sample)

            elif splt == "test":
                img_path, gt_path = self.imgs_labeled[index]  
                img_name = (img_path.split("/")[-1]).split('.')[0] 
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1
                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)             
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32)//225
                sample =  {'image': img, "classify": label, "name": img_name, 'SSL_S':gt}   
                if self.transform is not None:
                    sample= self.transform(sample)
     
                if self.classify:
                    sample = {'image': img, "classify": label, "name": img_name}
                elif self.SSL:
                    sample = {'image': img, 'SSL': gt, "name": img_name}
                    sample["idx"] = index 

        return sample

    def __len__(self):

        return self.length

    def __str__(self):
        return 'RCC_MT(split=' + str(self.split) + ')'


def test_all():
    import matplotlib.pyplot as plt
    import torch
    import data.custom_transforms_classify as tr
    from torchvision import transforms
    from utils.custom_collate import collate_mil

    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                                    flagvals={'image': cv2.INTER_NEAREST,
                                                              'classify': cv2.INTER_NEAREST}),
                                    tr.FixedResize(resolutions={'image': (512, 512),
                                                                'classify': (512, 512)},
                                                   flagvals={'image': cv2.INTER_NEAREST,
                                                             'classify': cv2.INTER_NEAREST}),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = RCC_context_classify(split='train', transform=transform, retname=True,
                            do_classify=True,
                            do_SSL=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
        print(sample["classify"].shape)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(1, 2)

            for k in range(len(ax_arr)):
                for l in range(len(ax_arr[k])):
                    ax_arr[k][l].cla()

            ax_arr[0][0].imshow(np.transpose(sample['image'][j], (1, 2, 0)))

            plt.show()
        break


if __name__ == '__main__':
    test_all()
