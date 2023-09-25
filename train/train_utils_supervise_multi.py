import argparse
from torchvision import transforms
from evaluation.evaluate_utils_supervise_multi import PerformanceMeter
from utils.utils_supervise_multi import AverageMeter, ProgressMeter, get_output
import torch
from utils import ramps
from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')

args = parser.parse_args()

        
def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = p.ALL_TASKS.NAMES
    tasks = p.TASKS.NAMES


    if p['model'] == 'mti_net': # Extra losses at multiple scales
        losses = {}
        for scale in range(4):
            for task in all_tasks:
                losses['scale_%d_%s' %(scale, task)] = AverageMeter('Loss scale-%d %s ' %(scale+1, task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    elif p['model'] == 'pad_net': # Extra losses because of deepsupervision
        losses = {}
        for task in all_tasks:
            losses['deepsup_%s' %(task)] = AverageMeter('Loss deepsup %s' %(task), ':.4e')
        for task in tasks:
            losses[task] = AverageMeter('Loss %s' %(task), ':.4e')


    else: # Only losses on the main task.
        losses = {task: AverageMeter('Loss %s' %(task), ':.4e') for task in tasks}


    losses['total'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(p, train_loader, model, ema_model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    criterion_SSL_S=criterion[0]
#    criterion_SSL_T=criterion[1]
    criterion_classify=criterion[1]
    
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))    
    
    model.train()
#    iter_num = 0
    tasks = p.ALL_TASKS.NAMES
#    print("tasks:",tasks)
    
    for i, batch in enumerate(train_loader):

        if "classify" in tasks:
            images = batch['image'].cuda()
            target_label = batch['classify'].cuda()    
            
            
        if "SSL_S" in tasks:
#            print(batch.keys())
            images= batch['image'].cuda()   
            targets = batch['SSL_S'].cuda()
         
        output = model(images)

        loss_SSL = criterion_SSL_S(output["SSL_S"], targets.long())
        
        loss_classify = criterion_classify(output["classify"], target_label)  
               
        loss_dict = {"SSL_S": loss_SSL, "classify":loss_classify, 'total': (loss_SSL*2 + loss_classify)/3}           
        
        
        for k, v in loss_dict.items():
            losses[k].update(v.item())
     
        output_dict={}
        targets_dict={}
        if "SSL_S" in tasks:
            output_dict["SSL_S"] = output["SSL_S"]
            targets_dict["SSL_S"] = targets
        if "classify" in tasks:
            output_dict["classify"] = output["classify"]
            targets_dict["classify"] = target_label
              
        performance_meter.update(output_dict, targets_dict)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
        
        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose=True)

    return eval_results
