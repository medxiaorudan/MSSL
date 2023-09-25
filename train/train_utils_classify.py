import argparse
from evaluation.evaluate_utils_classify import PerformanceMeter
from utils.utils import AverageMeter, ProgressMeter, get_output
import torch
from utils import ramps
from termcolor import colored

        
def get_loss_meters(p):
    """ Return dictionary with loss meters to monitor training """
    all_tasks = p.ALL_TASKS.NAMES
    tasks = p.TASKS.NAMES
    print("all_tasks",all_tasks)
    print("tasks",tasks)

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

#    print("losses:",losses)
    losses['classify'] = AverageMeter('Loss Total', ':.4e')
    return losses

def train_vanilla(p, train_loader, model, ema_model, criterion, optimizer, epoch):
    """ Vanilla training with fixed loss weights """

    criterion_classify=criterion
    
    losses = get_loss_meters(p)
    performance_meter = PerformanceMeter(p)
    progress = ProgressMeter(len(train_loader),
        [v for v in losses.values()], prefix="Epoch: [{}]".format(epoch))    
    
    model.train()
    iter_num = 0
    tasks = ["classify"]
    
    for i, batch in enumerate(train_loader):

        images = batch['image'].cuda()
        target_label = batch['classify'].cuda()    
                    
        output = model(images)
      
        loss_classify = criterion_classify(output, target_label)  
               
        loss_dict = {"classify":loss_classify}           
        
        
        for k, v in loss_dict.items():
            losses[k].update(v.item())
            
        output_dict={}
        targets_dict={}

        if "classify" in tasks:
            output_dict["classify"] = output
            targets_dict["classify"] = target_label
        
         
        performance_meter.update(output_dict, targets_dict)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['classify'].backward()
        optimizer.step()

#        if "SSL_T" in tasks:
#            #print(model.decoders)
#            update_ema_variables(model, None, args.ema_decay, iter_num)
        iter_num = iter_num + 1
        
        if i % 25 == 0:
            progress.display(i)

    eval_results = performance_meter.get_score(verbose=True)

    return eval_results
