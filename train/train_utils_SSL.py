
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
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=30.0, help='consistency_rampup')
args = parser.parse_args()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(model.module.decoders['SSL_T'].parameters(), model.module.decoders['SSL_S'].parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
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


    losses['SSL_S'] = AverageMeter('Loss Total', ':.4e')
    return losses


def train_vanilla(p, train_loader, model, ema_model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """
    criterion_SSL_S=criterion[0]
    criterion_SSL_T=criterion[1]
#    criterion_classify=criterion[2]
    
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))    
    
    model.train()
    iter_num = 0
    tasks = p.ALL_TASKS.NAMES
#    print("tasks:",tasks)
    to_pil_image = transforms.ToPILImage()
    cnt = 0      
    for i, batch in enumerate(train_loader):
      
        images= batch['image'].cuda()   
        targets = batch['SSL_S'].cuda()

        unlabeled_images = images[args.labeled_bs:]
        noise = torch.clamp(torch.randn_like(unlabeled_images) * 0.1, -0.2, 0.2)
        ema_inputs = unlabeled_images + noise                
            
        output = model(images)

        outputs_soft = torch.softmax(output['SSL_S'], dim=1)

        #with torch.no_grad():
            #ema_output = ema_model(ema_inputs)
        ema_output_soft = torch.softmax(output['SSL_T'], dim=1)        
    
        loss_ce = criterion_SSL_S(output['SSL_S'][:args.labeled_bs],
                          targets[:args.labeled_bs].long())
        loss_dice = criterion_SSL_T(outputs_soft[:args.labeled_bs], 
                          targets[:args.labeled_bs].long())
        supervised_loss = 0.5 * (loss_dice + loss_ce)
        consistency_weight = get_current_consistency_weight(iter_num//15)
        if iter_num < 100:
            consistency_loss = 0.0
        else:
            consistency_loss = torch.mean(
                (outputs_soft[args.labeled_bs:]-ema_output_soft)**2)
        loss_SSL = supervised_loss + consistency_weight * consistency_loss
               
        loss_dict = {"SSL_S": loss_SSL}           
        
        
        for k, v in loss_dict.items():
            losses[k].update(v.item())
        
        output_dict={}
        targets_dict={}
        if "SSL_S" in tasks:
            output_dict["SSL_S"] = output["SSL_S"]
            targets_dict["SSL_S"] = targets

         
        performance_meter.update(output_dict, targets_dict)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['SSL_S'].backward()
        optimizer.step()

        if "SSL_T" in tasks:
            #print(model.decoders)
            update_ema_variables(model, None, args.ema_decay, iter_num)
        iter_num = iter_num + 1
        
        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose=True)

    return eval_results
