import argparse
from torchvision import transforms
from evaluation.evaluate_utils_SSL import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output
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
parser.add_argument('--labeled_bs', type=int, default=3,
                    help='labeled_batch_size per gpu')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=30.0, help='consistency_rampup')
args = parser.parse_args()


def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = p.ALL_TASKS.NAMES
    tasks = ["SSL_S"]


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


    losses['SSL_S'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(p, train_loader, model, ema_model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    criterion_SSL_S=criterion[0]

#    criterion_classify=criterion[2]
    print(criterion_SSL_S)
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))    
    
    model.train()
#    iter_num = 0
    tasks = ["SSL_S"]
#    print("tasks:",tasks)

    to_pil_image = transforms.ToPILImage()
    cnt = 0    
    for i, batch in enumerate(train_loader):
     
        images= batch['image'].cuda()   
        targets = batch['SSL_S'].cuda()
          
        output = model(images)

        loss_SSL = criterion_SSL_S(output["SSL_S"], targets.long())
        print("loss_SSL:", loss_SSL)
        
        loss_dict = {"SSL_S": loss_SSL}                   
        
        for k, v in loss_dict.items():

            losses[k].update(v.item())
            
#        print(colored(targets.type,"yellow"))
        performance_meter.update({t: get_output(output[t], t) for t in ["SSL_S"]}, 
                                 {t: targets for t in ["SSL_S"]})        
        
        # Backward
        optimizer.zero_grad()
        loss_dict['SSL_S'].backward()
        optimizer.step()

        
        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose=True)

    return eval_results
