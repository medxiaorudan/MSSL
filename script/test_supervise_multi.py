import os
import json
import scipy.io as sio
import imageio
import torch
from PIL import Image
from torchvision import transforms
import argparse
import cv2
import os
import numpy as np
import sys
import torch
from utils.utils import get_output, mkdir_if_missing

from utils.config_SSL import create_config
from utils.common_config import get_transformations, \
                                get_test_dataset, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion_rudan
from evaluation.eval_SSL import eval_SSL
from evaluation.eval_classify import eval_classify

from utils.logger import Logger
#from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results
from termcolor import colored
parser = argparse.ArgumentParser(description='Vanilla Training')
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

def eval_all_results(p):
    """ Evaluate results for every task by reading the predictions from the save dir """
    save_dirs = os.path.join(p['save_dir'], 'test_result') 
    print("save_dirs:",save_dirs)
    results = {}

    if 'classify' in p.TASKS.NAMES: 
#        from evaluation.eval_classify import eval_classify_predictions
        results['classify'] = eval_classify_predictions(database=p['test_db_name'],
                             save_dir=save_dirs)
#        print("results['classify']:",results['classify'])
    
    if 'SSL_S' in p.TASKS.NAMES:
#        from evaluation.eval_SSL import eval_SSL_predictions
        results['SSL_S'] = eval_SSL_predictions(database=p['test_db_name'],
                              save_dir=save_dirs)

    if p['setup'] == 'multi_task': # Perform the multi-task performance evaluation
        single_task_test_dict = {}
        for task, test_dict in p.TASKS.SINGLE_TASK_TEST_DICT.items():
            with open(test_dict, 'r') as f_:
                 single_task_test_dict[task] = json.load(f_)
        print("results:",results)
#        print("single_task_test_dict:", single_task_test_dict)
        results['multi_task_performance'] = calculate_multi_task_performance(results, single_task_test_dict)   
   
        print('Multi-task learning performance on test set is %.2f' %(100*results['multi_task_performance']))

    return results


def eval_SSL_predictions(database, save_dir, overfit=False):
    """ Evaluate the segmentation maps that are stored in the save dir """

    # Dataloaders
    if database == 'RCCContext':
        from data.RCC_context import RCCContext
        n_classes = 1
#        cat_names = VOC_CATEGORY_NAMES
        has_bg = True
        gt_set = 'test'
        db = RCCContext(split=gt_set, do_classify=False, do_SSL=True,
                                          overfit=overfit)
  
    
    else:
        raise NotImplementedError
        
    base_name = database + '_' + 'test' + '_SSL_S'
    fname = os.path.join(save_dir, base_name + '.json')
    # Eval the model
    print('Evaluate the saved images (SSL_S)')
    eval_results = eval_SSL(db, os.path.join(save_dir, 'SSL_S'), n_classes=1, has_bg=True)
    with open(fname, 'w') as f:
        json.dump(eval_results, f)
        
    # Print results
    class_IoU = eval_results['jaccards_all_categs']
    mIoU = eval_results['mIoU']

    print('\nSegmentation mIoU: {0:.4f}\n'.format(100 * mIoU))


    return eval_results

def eval_classify_predictions(database, save_dir, overfit=False):
    """ Evaluate the segmentation maps that are stored in the save dir """
#    print("save_dir:",save_dir)
    # Dataloaders
    if database == 'RCCContext':
        from data.RCC_context import RCCContext
        n_classes = 2
        gt_set = 'test'
        
        db= RCCContext(split=gt_set, do_classify=True, do_SSL=False)
                

    else:
        raise NotImplementedError
    
    base_name = database + '_' + 'test' + '_classify'
    fname = os.path.join(save_dir, base_name + '.json')

    # Eval the model
    print('Evaluate the saved images (classify)')
    
    eval_results = eval_classify(db, os.path.join(save_dir, 'classify'))
    

    
    with open(fname, 'w') as f:
        json.dump(eval_results, f)
        
    # Print results

    print('Results for Classify')
    for i in eval_results:
        spaces = ''
        for j in range(0, 15 - len(i)):
            spaces += ' '
        print('{0:s}{1:s}{2:.4f}'.format(i, spaces, eval_results[i]))

    return eval_results
        


def main():
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms, val_transforms = get_transformations(p)
    
    print(val_transforms)
    
#    test_dataset = get_test_dataset(p, train_transforms)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_val_dataloader(p, test_dataset)
#    for i in test_dataloader:
#        print(i["name"])
    model = get_model(p)  
    model = torch.nn.DataParallel(model)
#    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.cuda()

    # load model weights
    weights_path = "/data/morpheme/user/rxiao/MIDL/deep_learning/Multi-task/results_supervise_multi/RCCContext/hrnet_w18/multi_task_baseline/best_model.pth.tar"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        tasks = p.TASKS.NAMES
        save_dirs = {task: os.path.join(p['save_dir'], 'test_result/'+ task) for task in tasks}       
#        save_dirs = os.path.join(p['save_dir'], 'test_result') 
        for save_dir in save_dirs.values():
            mkdir_if_missing(save_dir)        

        for ii, sample in enumerate(test_dataloader): 
            print(sample["name"])
            if "SSL_S" in tasks:
                inputs, targets, name = sample['image'].cuda(), sample['SSL_S'].cuda(), sample['name']
                img_size = (inputs.size(2), inputs.size(3))
    #            inputs = inputs.permute(0,3,1,2)
                output = model(inputs)
    #            print("output.shape:",output["SSL_S"].shape)
                output_task = get_output(output["SSL_S"], "SSL_S").cpu().data.numpy()
    #            print(output_task.keys())
    #            output_task = get_output(output, "SSL").cpu().data.numpy()
                for jj in range(int(inputs.size()[0])):
    #                print(inputs.size()[0])
    #                print(output_task[0][:][-2].shape)
    #                print(targets.size()[1])
    #                print(targets.size()[2])
                    fname = name[jj]  
    #                result = cv2.resize(output_task[jj][:][-2], dsize=(targets.size()[1], targets.size()[2]),  interpolation=p.TASKS.INFER_FLAGVALS['SSL_S'])  #problems
                    result = output_task[jj]
                    imageio.imwrite(os.path.join(save_dirs["SSL_S"], fname + '.png'), result)
    
              
            if "classify" in tasks:
                inputs, targets, name = sample['image'].cuda(), sample['classify'].cuda(), sample['name']
    #            img_size = (inputs.size(2), inputs.size(3))
    #            inputs = inputs.permute(0,3,1,2)
    #            print("*******************")
    #            print(inputs)
    #            print("*******************")
                output = model(inputs)            
                output_task = get_output(output["classify"], "classify").cpu().data.numpy()
    #            print(output_task)
    #            print("&&&&&&&&&&&&&&&&&&&")
    #            print(output)
                print(output_task)
    #            print(save_dirs)
    #            print("&&&&&&&&&&&&&&&&&&&")
    #            final_model_state_file = os.path.join(save_dirs["classify"],
    #                                          'final_state.pth.tar')
    ##            torch.save(model.module.state_dict(), final_model_state_file)
                for jj in range(int(inputs.size()[0])):
                   fname = name[jj] 
                   sio.savemat(os.path.join(save_dirs["classify"], fname + '.mat'), {'classify': output_task})    
        eval_stats = eval_all_results(p)    
#    
#    
#    
#    correct_1 = 0.0
#    correct_2 = 0.0
#    with torch.no_grad():
#        # predict class
#        for n_iter, batch in enumerate(test_dataloader):
##            image = image.cuda()
##            label = label.cuda()
#            image = batch['image'].to(device)
#            label = batch['classify'].to(device)
##            output = model(image.to(device)).cpu()
##            predict = torch.softmax(output, dim=1)
##            probs, classes = torch.max(predict, dim=1)
#            output = model(image)
#            _, pred = output.topk(2, 1, largest=True, sorted=True)
#            label = label.view(label.size(0), -1).expand_as(pred)
#            correct = pred.eq(label).float()        
#            correct_1 += correct[:, :1].sum()
#            correct_2 += correct[:, :2].sum()
#        print("ccRCC_correct: ",correct_1.cpu().numpy(),"/"+str(len(test_dataset)))
#        print("pRCC_correct: ",correct_2.cpu().numpy(),"/"+str(len(test_dataset)))
#        print("ccRCC err: ", 1 - correct_1 / len(test_dataset))
#        print("pRCC err: ", 1 - correct_2 / len(test_dataset))
#    #        for idx, (pro, cla) in enumerate(zip(probs, classes)):
#    #            print("image: {}  class: {}  prob: {:.3}".format(img_path_list[idx],
    #                                                             class_indict[str(cla.numpy())],
    #                                                             pro.numpy()))


if __name__ == '__main__':
    main()

