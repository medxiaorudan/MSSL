import os
import json

import torch
from PIL import Image
from torchvision import transforms
import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config import create_config
from utils.common_config_classify import get_transformations, \
                                get_test_dataset, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion_rudan
from utils.logger import Logger
#from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions, \
                                    eval_all_results
from termcolor import colored
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()
def main():
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms, val_transforms = get_transformations(p)
    
    print(val_transforms)
    
#    test_dataset = get_test_dataset(p, train_transforms)
    test_dataset = get_test_dataset(p, val_transforms)
    test_dataloader = get_val_dataloader(p, test_dataset)
    
    model = get_model(p)  
    model = torch.nn.DataParallel(model)
#    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.cuda()

    # load model weights
    weights_path = "/data/morpheme/user/rxiao/MIDL/deep_learning/Multi-task/results_classify/RCCContext_classify/hrnet_w18/single_task/classify/best_model.pth.tar"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    correct_1 = 0.0
    correct_2 = 0.0
    with torch.no_grad():
        # predict class
        for n_iter, batch in enumerate(test_dataloader):

            image = batch['image'].to(device)
            label = batch['classify'].to(device)

            output = model(image)
            _, pred = output.topk(2, 1, largest=True, sorted=True)
            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()        
            correct_1 += correct[:, :1].sum()
            correct_2 += correct[:, :2].sum()
        print("ccRCC_correct: ",correct_1.cpu().numpy(),"/"+str(len(test_dataset)))
        print("pRCC_correct: ",correct_2.cpu().numpy(),"/"+str(len(test_dataset)))
        print("ccRCC err: ", 1 - correct_1 / len(test_dataset))
        print("pRCC err: ", 1 - correct_2 / len(test_dataset))


if __name__ == '__main__':
    main()
