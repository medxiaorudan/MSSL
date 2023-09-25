import warnings
import cv2
import os.path
import glob
import json
import numpy as np
import torch
from PIL import Image
import evaluation.jaccard as evaluation
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io, transform

VOC_CATEGORY_NAMES = ['background',
                      'vascular']



def eval_SSL(loader, folder, n_classes=1, has_bg=True):


    eval_result = dict()

    n_classes = n_classes + int(has_bg)
    # Iterate

    jac=[]
    prec=[]
    rec=[]
    dice=[]
    for i, sample in enumerate(loader):
        if i % 10 == 0:        
            print('Evaluating: {} of {} objects'.format(i, len(loader)))

        # Load result
        filename = os.path.join(folder, sample['name'] + '.png')

        mask = np.array(Image.open(filename).convert('L')).astype(np.float32)//255

        gt = sample['SSL_S']

        if mask.shape != gt.shape:
            warnings.warn('Prediction and ground truth have different size. Resizing Prediction..')

            gt = Image.fromarray(gt)
            resize = transforms.Resize([512,512])
            gt = np.array(resize(gt).convert('L')).astype(np.float32)

        jac.append(evaluation.jaccard(gt, mask))

        prec_i, rec_i = evaluation.precision_recall(gt, mask)
        prec.append(prec_i)
        rec.append(rec_i)
        dice.append(evaluation.dice_coeff(mask,gt))
        

    # Write results
    
    eval_result['jaccards_all_categs'] = [0] * n_classes
    eval_result['mIoUs'] = np.mean(jac)
    eval_result['mPrec'] = np.mean(rec)   
    eval_result['mRec'] = np.mean(prec)
    eval_result['mDices'] = np.mean(dice)
        
    eval_result['mIoU'] = np.max(jac)
    eval_result['mDice'] = np.max(dice)


      
    print(eval_result)    
    
    return eval_result


class SSLMeter(object):
    def __init__(self, database):
        if database == 'RCCContext':
            n_classes = 1
            cat_names = VOC_CATEGORY_NAMES
            has_bg = True
                  
        else:
            raise NotImplementedError
        
        self.n_classes = n_classes + int(has_bg)
#        self.cat_names = cat_names
        self.jac=[]
        self.prec=[]
        self.rec=[]
        self.dice=[]

        

    @torch.no_grad()
    def update(self, pred, gt):
        
        
        if gt.shape!=pred.shape:
            pred=pred.permute(1,0,2,3)[1,:]

        self.jac.append(evaluation.jaccard(gt, pred))

        self.prec_i, self.rec_i = evaluation.precision_recall(gt, pred)
        self.prec.append(self.prec_i)
        self.rec.append(self.rec_i)
        self.dice.append(evaluation.dice_coeff(pred,gt))
        
    def reset(self):
        self.jac=[]
        self.prec=[]
        self.rec=[]
        self.dice=[]
            
    def get_score(self, verbose=True):

        eval_result = dict()
        eval_result['jaccards_all_categs'] = [0] * self.n_classes
        eval_result['mIoUs'] = np.mean(self.jac)
        eval_result['mPrec'] = np.mean(self.rec)   
        eval_result['mRec'] = np.mean(self.prec)
        eval_result['mDices'] = np.mean(self.dice)
        eval_result['mIoU'] = np.max(self.jac)
        eval_result['mDice'] = np.max(self.dice)
        
        if verbose:

            pass
        return eval_result


def eval_SSL_predictions(database, save_dir, overfit=False):
    """ Evaluate the segmentation maps that are stored in the save dir """

    # Dataloaders
    if database == 'RCCContext':
        from data.RCC_context import RCCContext
        n_classes = 1
        cat_names = VOC_CATEGORY_NAMES
        has_bg = True
        gt_set = 'val'
        db = RCCContext(split=gt_set, do_classify=False, do_SSL=True,
                                          overfit=overfit)
   
    else:
        raise NotImplementedError
        
    base_name = database + '_' + 'test' + '_SSL_S'
    fname = os.path.join(save_dir, base_name + '.json')

    print('Evaluate the saved images (SSL_S)')
    eval_results = eval_SSL(db, os.path.join(save_dir, 'SSL_S'), n_classes=1, has_bg=True)
    with open(fname, 'w') as f:
        json.dump(eval_results, f)
        
    # Print results
    class_IoU = eval_results['jaccards_all_categs']
    mIoU = eval_results['mIoU']

    print('\nSegmentation mIoU: {0:.4f}\n'.format(100 * mIoU))

    return eval_results
