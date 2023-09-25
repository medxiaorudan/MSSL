import os
import sys
import tarfile
import json
import cv2
from torchvision import transforms
import numpy as np
import scipy.io as sio
import torch.utils.data as data
from PIL import Image, ImageFile
from skimage.morphology import thin
from six.moves import urllib
import torch
from utils.mypath import MyPath, PROJECT_ROOT_DIR
import matplotlib.pyplot as plt
import math

from utils.common_config import get_transformations

ImageFile.LOAD_TRUNCATED_IMAGES = True

class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("please set size as int or tuple")

    def __call__(self, img, mask):
        img = img.resize(self.size)
        mask = mask.resize(self.size)
        return img, mask

def make_dataset(root, prefix=('jpg', 'png')):
    img_label_path = root[0]
    gt_label_path = root[1]
    img_list = [os.path.splitext(f)[0] for f in os.listdir(img_label_path) if f.endswith(prefix[1])]
    return [(os.path.join(img_label_path, img_name + prefix[1]), os.path.join(gt_label_path, img_name + prefix[1])) for img_name in img_list]

class RCCContext(data.Dataset):
    """
    PASCAL-Context dataset, for multiple tasks
    Included tasks:
        1. Edge detection,
        2. Semantic Segmentation,
        3. Human Part Segmentation,
        4. Surface Normal prediction (distilled),
        5. Saliency (distilled)
    """

    VOC_CATEGORY_NAMES = [
                          'vascular']

    CONTEXT_CATEGORY_LABELS = [0,
                               1]

    def __init__(self,
                 root='/data/morpheme/user/rxiao/MIDL/bigdata/',
                 download=False,
                 split='val',
                 in_size=512,
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
        self.train_joint_transform = JointResize(in_size)
        self.transform = transform
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.area_thres = area_thres
        self.retname = retname

        # Edge Detection
        self.classify = do_classify
        self.SSL = do_SSL
        self.train_labeled_path = [os.path.join(self.root, 'train', 'img_labeled'),os.path.join(self.root, 'train', 'seg')]

        self.train_unlabeled_path = [os.path.join(self.root, 'train', 'img_unlabeled'),os.path.join(self.root, 'train', 'img_unlabeled')]

        self.val_path = [os.path.join(self.root, 'val', 'img_labeled'),os.path.join(self.root, 'val', 'seg')]
        self.test_path = [os.path.join(self.root, 'test', 'img_labeled'),os.path.join(self.root, 'test', 'seg')]

        print("Initializing dataloader for RCC {} set".format(''.join(self.split)))
        for splt in self.split:        
        
            if splt == "train":
                
                self.imgs_labeled = make_dataset(self.train_labeled_path, prefix=('.jpg', '.png'))
                
                len_labeled = len(self.imgs_labeled)    
        
                self.imgs_unlabeled = make_dataset(self.train_unlabeled_path, prefix=('.jpg', '.png'))
                
                len_unlabeled = self.r_l_rate * len_labeled
#                len_unlabeled = len(self.imgs_unlabeled)
                
                self.length = len_labeled + len_unlabeled
                
                for i in self.imgs_unlabeled:
                    self.imgs_labeled.append(i)
                
                print('Number of dataset images: {:d}'.format(self.length))
    
            elif splt == "val":
            
                self.imgs_labeled = make_dataset(self.val_path, prefix=('.jpg', '.png'))
                len_labeled = len(self.imgs_labeled)

                self.length = len_labeled

            elif splt == "test":
            
                self.imgs_labeled = make_dataset(self.test_path, prefix=('.jpg', '.png'))

                len_labeled = len(self.imgs_labeled)
                self.length = len_labeled
       
            print('Number of dataset images: {:d}'.format(self.length))
            
        
    def __getitem__(self, index):
        sample = {}

        for splt in self.split: 
                      
            if splt == "train":
                img_path, gt_path = self.imgs_labeled[index]        
                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)  
                img_name = (img_path.split("/")[-1]).split('.')[0]        
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1   
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32)//225
                sample =  {'image': img, "classify": label, "name": img_name, 'SSL_S':gt}
                if self.transform is not None:
                    sample = self.transform(sample)
                                         
            elif splt == "val":
                img_path, gt_path = self.imgs_labeled[index]  
                img_name = (img_path.split("/")[-1]).split('.')[0] 
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1
#                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
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

        return sample

    def __len__(self):

        return self.length

    def __str__(self):
        return 'RCC_MT(split=' + str(self.split) + ')'


class RCCContext_supervise(data.Dataset):


    VOC_CATEGORY_NAMES = [
                          'vascular']

    CONTEXT_CATEGORY_LABELS = [0,
                               1]

    def __init__(self,
                 root='/data/morpheme/user/rxiao/MIDL/bigdata/',
                 download=False,
                 split='train',
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
        self.train_path = [os.path.join(self.root, 'train', 'img_labeled'),os.path.join(self.root, 'train', 'seg')]
        self.val_path = [os.path.join(self.root, 'val', 'img_labeled'),os.path.join(self.root, 'val', 'seg')]


        print("Initializing dataloader for RCC {} set".format(''.join(self.split)))
        for splt in self.split:        
        
            if splt == "train":
                
                self.imgs_labeled = make_dataset(self.train_path, prefix=('.jpg', '.png'))
                
                len_labeled = len(self.imgs_labeled)    
                        
                self.length = len_labeled
                    
            if splt == "val":
            
                self.imgs_labeled = make_dataset(self.val_path, prefix=('.jpg', '.png'))
                
                len_labeled = len(self.imgs_labeled)
                self.length = len_labeled
#                                                        
            print('Number of dataset images: {:d}'.format(self.length))
            

    def __getitem__(self, index):
        sample = {}

        for splt in self.split: 
                        
            if splt == "train":
                img_path, gt_path = self.imgs_labeled[index]

                img_name = (img_path.split("/")[-1]).split('.')[0]  
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1    
                        
                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)             
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32)//225
                
                sample =  {'image': img, "classify": label, "name": img_name, 'SSL_S':gt}

                if self.transform is not None:
                    sample= self.transform(sample)

                                          
            if splt == "val":
                img_path, gt_path = self.imgs_labeled[index]  

                img_name = (img_path.split("/")[-1]).split('.')[0] 
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1

                img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)             
                gt = np.array(Image.open(gt_path).convert('L')).astype(np.float32)//225

                sample =  {'image': img, "classify": label, "name": img_name, 'SSL_S':gt}  

                if self.transform is not None:
                    sample= self.transform(sample)  

        return sample

    def __len__(self):

        return self.length


#
    def __str__(self):
        return 'RCC_MT(split=' + str(self.split) + ')'


class RCCContext_classify(data.Dataset):

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

        self.train_unlabeled_path = [os.path.join(self.root, 'train', 'img'),os.path.join(self.root, 'train', 'img')]

        self.val_path = [os.path.join(self.root, 'val', 'img'),os.path.join(self.root, 'val', 'img')]


        print("Initializing dataloader for RCC {} set".format(''.join(self.split)))
        for splt in self.split:        
        
            if splt == "train":
                
        
                self.imgs_unlabeled = make_dataset(self.train_unlabeled_path, prefix=('.jpg', '.png'))

                len_unlabeled = len(self.imgs_unlabeled)
                
                self.length = len_unlabeled
                 
            if splt == "val":
            
                self.imgs_unlabeled = make_dataset(self.val_path, prefix=('.jpg', '.png'))
                len_unlabeled = len(self.imgs_unlabeled)

                self.length = len_unlabeled
            
            print('Number of dataset images: {:d}'.format(self.length))


    def __getitem__(self, index):
        sample = {}

        for splt in self.split: 
                
            if splt == "train":
                img_path, gt_path = self.imgs_unlabeled[index]  # 0, 1 => 10550
                                    
                img_name = (img_path.split("/")[-1]).split('.')[0]  
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1    
                         
                img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)
                sample =  {'image': img, "classify": label, "name": img_name}

                if self.transform is not None:
                    sample= self.transform(sample)
                                          
            if splt == "val":
                img_path, gt_path = self.imgs_unlabeled[index]  
                img_name = (img_path.split("/")[-1]).split('.')[0] 
                label = 0 if img_name.split("_")[0] ==  "ccRCC" else 1
                img = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32)

                sample =  {'image': img, "classify": label, "name": img_name}  

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
        return 'RCC_MT(split=' + str(self.split) + '+"classify_val")'

def test_all():
    import matplotlib.pyplot as plt
    import torch
    import data.custom_transforms as tr
    from torchvision import transforms
    from utils.custom_collate import collate_mil

    transform = transforms.Compose([tr.RandomHorizontalFlip(),
                                    tr.ScaleNRotate(rots=(-90, 90), scales=(1., 1.),
                                                    flagvals={'image': cv2.INTER_NEAREST,
                                                              'SSL': cv2.INTER_NEAREST
                                                              }),
                                    tr.FixedResize(resolutions={'image': cv2.INTER_NEAREST,
                                                                'SSL': (512, 512)
                                                                },
                                                   flagvals={'image': cv2.INTER_NEAREST,
                                                             'SSL': cv2.INTER_NEAREST
                                                             }),
                                    tr.AddIgnoreRegions(),
                                    tr.ToTensor()])
    dataset = RCCContext(split='train', transform=transform, retname=True,
                            do_classify=True,
                            do_SSL=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    for i, sample in enumerate(dataloader):
#        print(i)
        for j in range(sample['image'].shape[0]):
            f, ax_arr = plt.subplots(1, 2)

            for k in range(len(ax_arr)):
                for l in range(len(ax_arr[k])):
                    ax_arr[k][l].cla()

            ax_arr[0][0].imshow(np.transpose(sample['image'][j], (1, 2, 0)))
            ax_arr[0][1].imshow(np.transpose(sample['SSL'][j], (1, 2, 0))[:, :, 0])

            plt.show()
        break


if __name__ == '__main__':
    test_all()
