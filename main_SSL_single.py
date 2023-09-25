#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config_SSL import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion_rudan
from utils.logger import Logger
from train.train_utils_SSL import train_vanilla
from evaluation.evaluate_utils_SSL import eval_model, validate_results, save_model_predictions, \
                                    eval_all_results
from termcolor import colored
from torchvision import models
from torchsummary import summary
from torchvision import transforms
# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def main():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
#    print("###################################")
##    print(p)
#    print("###################################")
    model = get_model(p)
        
    #ema_model = get_model(p)
    
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    summary(model,(3, 512,512))
    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion_classify = get_criterion_rudan(p,"classify").cuda()
    criterion_SSL_S = get_criterion_rudan(p,"SSL_S").cuda()
    criterion_SSL_T = get_criterion_rudan(p,"SSL_T").cuda()
    criterion=[criterion_SSL_S,criterion_SSL_T,criterion_classify]

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
#    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
#    train_joint_transform = transforms.Compose([
#                                   transforms.ToTensor()])
    train_dataset = get_train_dataset(p,train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
#    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    train_dataloader = get_train_dataloader(p, train_dataset,task='SSL_S')
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)
    
    # Resume from checkpoint
#    if os.path.exists(p['checkpoint']):
#        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
#        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
#        optimizer.load_state_dict(checkpoint['optimizer'])
#        model.load_state_dict(checkpoint['model'])
#        start_epoch = checkpoint['epoch']
#        best_result = checkpoint['best_result']

#    else:
    print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
    start_epoch = 0
    save_model_predictions(p, val_dataloader, model)
    best_result = eval_all_results(p)
#    best_result = 0    
    start_epoch = 0
    # Main loop
    print(colored('Starting main loop', 'blue'))

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

#        print(p)
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        
#        print("criterion:", criterion)
        
        eval_train = train_vanilla(p, train_dataloader, model, None, criterion, optimizer, epoch)
        
        # Evaluate
            # Check if need to perform eval first
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
            if epoch + 1 > p['epochs'] - 10:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = True

        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions(p, val_dataloader, model)
            curr_result = eval_all_results(p)
            improves, best_result = validate_results(p, curr_result, best_result, "SSL_S")
            if improves:
                print('Save new best model')
                torch.save(model.state_dict(), p['best_model'])

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])

    # Evaluate best model at the end
    print(colored('Evaluating best model at the end', 'blue'))
    model.load_state_dict(torch.load(p['checkpoint'])['model'])
    save_model_predictions(p, val_dataloader, model)
    eval_stats = eval_all_results(p)

if __name__ == "__main__":
    main()
