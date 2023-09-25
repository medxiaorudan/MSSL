import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils_classify import get_output, mkdir_if_missing
from evaluation.eval_classify_classify import accuracy, AverageMeter

class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p):
        self.database = "RCC_context_classify"
        print("self.database:", self.database)
#        self.tasks = p.TASKS.NAMES
        self.tasks = ["classify"]
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()
    
    def update(self, pred, gt):
        
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict


def calculate_multi_task_performance(eval_dict, single_task_dict):
    assert(set(eval_dict.keys()) == set(single_task_dict.keys()))
    tasks = eval_dict.keys()
    num_tasks = len(tasks)    
    mtl_performance = 0.0

    for task in tasks:
        mtl = eval_dict[task]
        stl = single_task_dict[task]
        
        if task == 'SSL_S': # rmse lower is better
            mtl_performance += (mtl['mIoU'] - stl['mIoU'])/stl['mIoU']

        elif task == 'classify': # mean error lower is better
            mtl_performance += (mtl['accuracy'] - stl['accuracy'])/stl['accuracy']


        else:
            raise NotImplementedError

    return mtl_performance / num_tasks



def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """

    if task == 'classify':
        from evaluation.eval_classify_classify import AverageMeter
        return AverageMeter()
        

    else:
        raise NotImplementedError


def validate_results(p, current, reference):
    """
        Compare the results between the current eval dict and a reference eval dict.
        Returns a tuple (boolean, eval_dict).
        The boolean is true if the current eval dict has higher performance compared
        to the reference eval dict.
        The returned eval dict is the one with the highest performance.
    """
    tasks = ["classify"]
    
    if len(tasks) == 1: # Single-task performance
        task = tasks[0]               
        
        if task == 'classify': 
            if current['classify']['accuracy'] > reference['classify']['accuracy']:
                print('New best human parts semgentation model %.2f -> %.2f' %(100*reference['classify']['accuracy'], 100*current['classify']['accuracy']))
                improvement = True
            else:
                print('No new best human parts semgentation model %.2f -> %.2f' %(100*reference['classify']['accuracy'], 100*current['classify']['accuracy']))
                improvement = False        
        

    else: # Multi-task performance
        if current['multi_task_performance'] > reference['multi_task_performance']:
            print('New best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = True

        else:
            print('No new best multi-task model %.2f -> %.2f' %(100*reference['multi_task_performance'], 100*current['multi_task_performance']))
            improvement = False

    if improvement: # Return result
        return True, current

    else:
        return False, reference


@torch.no_grad()
def eval_model(p, val_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = ["classify"]
    performance_meter = PerformanceMeter(p)

    model.eval()

    for i, batch in enumerate(val_loader):
        # Forward pass
        images = batch['image'].cuda(non_blocking=True)
        print("images.shape:",images.shape)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}
        output = model(images)

        # Measure performance
        performance_meter.update({t: get_output(output[t], t) for t in tasks}, targets)

    eval_results = performance_meter.get_score(verbose = True)
    return eval_results


@torch.no_grad()
def save_model_predictions(p, val_loader, model):
    """ Save model predictions for all tasks """        
    print('Save model predictions to {}'.format(p['save_dir']))
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    
    model.eval()
    tasks = p.TASKS.NAMES
   
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks}
#    print(save_dirs)
#    print(p['save_dir'])
    
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
#    print(len(val_loader))
    for ii, sample in enumerate(val_loader): 
          
        if "classify" in tasks:
            inputs, targets, name = sample['image'].cuda(), sample['classify'].cuda(), sample['name']
            output = model(inputs)  
            pred = accuracy(output, targets, (1, ))
            final_model_state_file = os.path.join(save_dirs["classify"],
                                          'final_state.pth.tar')
            for jj in range(int(inputs.size()[0])):
               fname = name[jj] 
               sio.savemat(os.path.join(save_dirs["classify"], fname + '.mat'), {'classify': pred.item()})    


def eval_all_results(p):
    """ Evaluate results for every task by reading the predictions from the save dir """
    save_dir = p['save_dir'] 

    results = {}

    if 'classify' in p.TASKS.NAMES: 
        from evaluation.eval_classify_classify import eval_classify_predictions
        results['classify'] = eval_classify_predictions(database="RCC_context_classify",
                             save_dir=save_dir)

    if p['setup'] == 'multi_task': # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(test_dict, 'r') as f_:
                 single_task_test_dict[task] = json.load(f_)
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)            
        print('Multi-task learning performance on test set is %.2f' %(100*results['multi_task_performance']))

    return results
