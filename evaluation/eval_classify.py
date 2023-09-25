import torch
import warnings
import cv2
import os
import glob
import json
import numpy as np
import torch
from PIL import Image
import time
import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score

VOC_CATEGORY_NAMES = ['0',
                      '1']
                     
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = 1

        _, pred = output.topk(maxk, 1, True, True)
        
        pred = pred.t()

        return pred

        
def eval_classify(val_loader, 
             folder):
    print("folder", folder)
    # switch to evaluate mode
    acc=0
    val_num=0
    print("len(val_loader):",len(val_loader))
    for i, sample in enumerate(val_loader):

        if i % 10 == 0:
            print('Evaluating depth: {} of {} objects'.format(i, len(val_loader)))

        # Load result
        filename = os.path.join(folder, sample['name'] + '.mat')
        pred = sio.loadmat(filename)['classify'].astype(np.float32)

        print(pred)
        label = sample['classify']
        print(label)
        
#        print(torch.Tensor(np.array([label])))
        print("pred:", pred)
        acc += torch.eq(torch.Tensor(pred)[0][0], torch.Tensor(np.array([label]))).sum().item()
        val_num += 1        
    val_accurate = acc / val_num        
    # Write results
    eval_result = dict()
    eval_result['accuracy'] = val_accurate
    
    print(eval_result)
    
    return eval_result



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.pred_list = []
        self.gt_list = []
        self.count = 0

    def update(self, pred, gt):
#    def update(self, pred, gt):
        self.pred_list.append(pred)
        self.gt_list.append(gt)
        self.count += len(self.gt_list)        
   
#    def get_score(self):
    def get_score(self, metric='accuracy'):
        pred_array = torch.cat(self.pred_list).detach().cpu().numpy()
        gt_array = torch.cat(self.gt_list).detach().cpu().numpy()
        if metric == "accuracy":
            return accuracy_score(gt_array, pred_array)
        if metric == "f1_score":
            return f1_score(gt_array, pred_array, average="weighted")    
        
        
def eval_classify_predictions(database, save_dir, overfit=False):
    """ Evaluate the segmentation maps that are stored in the save dir """

    # Dataloaders
    if database == 'RCC_context_classify':
        from data.RCC_context_classify import RCC_context_classify
        n_classes = 2
#        cat_names = VOC_CATEGORY_NAMES
#        has_bg = True
        gt_set = 'val'
        
        db= RCC_context_classify(split=gt_set, do_classify=True, do_SSL=False)
                  
    else:
        raise NotImplementedError
    
    base_name = database + '_' + 'test' + '_classify'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (classify)')
    
    eval_results = eval_classify(db, os.path.join(save_dir, 'classify'))
    
    
    with open(fname, 'w') as f:
        json.dump(eval_results, f)
        
    print('Results for Classify')
    for i in eval_results:
        spaces = ''
        for j in range(0, 15 - len(i)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(i, spaces, eval_results[i]))

    return eval_results
        
        