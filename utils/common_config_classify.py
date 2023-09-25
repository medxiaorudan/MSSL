import os
import copy
import torch
import torch.nn.functional as F
import torch
from torchsummary import summary

from torchvision import transforms
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools



def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

device = torch.device("cuda")
"""
    Model getters 
"""
def iterate_once(iterable):
    return np.random.permutation(iterable)
    
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())
        
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size



def get_backbone(p):
    """ Return the backbone """

    if p['backbone'] == 'resnet18':
        from models.resnet import resnet18
        backbone = resnet18(p['backbone_kwargs']['pretrained'])
        backbone_channels = 512
    
    elif p['backbone'] == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(p['backbone_kwargs']['pretrained'])
        backbone_channels = 2048

    elif p['backbone'] == 'hrnet_w18':
        from models.seg_hrnet import hrnet_w18
#        params=p['backbone_kwargs']
        backbone = hrnet_w18(p['backbone_kwargs']['pretrained'])
        backbone_channels = [18, 36, 72, 144]
    
    else:
        raise NotImplementedError

    if p['backbone_kwargs']['dilated']: # Add dilated convolutions
        assert(p['backbone'] in ['resnet18', 'resnet50'])
        from models.resnet_dilated import ResnetDilated
        backbone = ResnetDilated(backbone)

    if 'fuse_hrnet' in p['backbone_kwargs'] and p['backbone_kwargs']['fuse_hrnet']: # Fuse the multi-scale HRNet features
        from models.seg_hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """ Return the decoder head """
    if task == 'classify':
        from models.seg_hrnet import HighResolutionHeadClassify
        
        #model= DeepLabHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
        model= HighResolutionHeadClassify(backbone_channels, p.TASKS.NUM_OUTPUT[task])
#        summary(model, (16, 2, 3, 512, 512))
#        model = GoogLeNet( num_classes=2)

    else:
        from models.seg_hrnet import HighResolutionHead
        model= HighResolutionHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
        
    ema = p.TASKS.EMA[task]
    if ema:
        for param in model.parameters():
            param.detach_()      
             
    return model
    
def get_model(p):
    """ Return the model """

    backbone, backbone_channels = get_backbone(p)
    
    if p['setup'] == 'single_task':
        from models.models_classify import SingleTaskModel
        task = p.TASKS.NAMES[0]
        head = get_head(p, backbone_channels, task)
        model = SingleTaskModel(backbone, head, task)


    elif p['setup'] == 'multi_task':
        if p['model'] == 'baseline':
            from models.models import MultiTaskModel
            heads = torch.nn.ModuleDict({task: get_head(p, backbone_channels, task) for task in p.TASKS.NAMES})
            model = MultiTaskModel(backbone, heads, p.TASKS.NAMES)

    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import custom_transforms_classify as tr


    if p['train_db_name'] == 'RCCContext_classify':
        # Horizontal flips with probability of 0.5
        transforms_tr = [tr.RandomHorizontalFlip()]
        
        # Rotations and scaling
        transforms_tr.extend([tr.ScaleNRotate(rots=(-20, 20), scales=(.75, 1.25),
                                              flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])

    else:
        raise ValueError('Invalid train db name'.format(p['train_db_name']))


    # Fixed Resize to input resolution
    transforms_tr.extend([tr.FixedResize(resolutions={x: tuple(p.TRAIN.SCALE) for x in p.ALL_TASKS.FLAGVALS},
                                         flagvals={x: p.ALL_TASKS.FLAGVALS[x] for x in p.ALL_TASKS.FLAGVALS})])
    transforms_tr.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
    transforms_tr = transforms.Compose(transforms_tr)

    
    # Testing (during training transforms)
    transforms_ts = []
    transforms_ts.extend([tr.FixedResize(resolutions={x: tuple(p.TEST.SCALE) for x in p.TASKS.FLAGVALS},
                                         flagvals={x: p.TASKS.FLAGVALS[x] for x in p.TASKS.FLAGVALS})])
    transforms_ts.extend([tr.AddIgnoreRegions(), tr.ToTensor()])
    transforms_ts = transforms.Compose(transforms_ts)

    return transforms_tr, transforms_ts


def get_train_dataset(p, transforms):
    """ Return the train dataset """

    db_name = "RCCContext_classify"
    print('Preparing train loader for db: {}'.format(db_name))
    from data.RCC_context_classify import RCC_context_classify
    database = RCC_context_classify(split=['train'], transform=transforms, retname=True,
                                      do_classify='classify' in p.ALL_TASKS.NAMES,
                                      do_SSL=False, split_rate=(1, 3),
                                      overfit=None)
    return database


def get_train_dataloader(p, dataset,task=None):
    """ Return the train dataloader """
    total_slices = len(dataset)    
    labeled_slice = 223
    for task in p.ALL_TASKS.NAMES:    
        if task == 'classify':
            trainloader = DataLoader(dataset, batch_size=p['trBatch'], shuffle=True, drop_last=True,
                                     num_workers=0, collate_fn=collate_mil)
#        if task == 'SSL_S':
#            trainloader = DataLoader(dataset, batch_sampler=batch_sampler, 
#                                     num_workers=0, pin_memory=True, collate_fn=collate_mil)                                 
                                 
    return trainloader


def get_val_dataset(p, transforms):
    """ Return the validation dataset """

    db_name = p['val_db_name']
    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'RCCContext_classify':
        from data.RCC_context_classify import RCC_context_classify
        database = RCC_context_classify(split=['val'], transform=transforms, retname=True,
                                      do_classify='classify' in p.TASKS.NAMES,
                                      do_SSL=False,
                                    overfit=None)
    
    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database

def get_test_dataset(p, transforms):
    """ Return the validation dataset """

    db_name = p['val_db_name']

    print('Preparing val loader for db: {}'.format(db_name))

    if db_name == 'RCCContext_classify':
        from data.RCC_context_classify import RCC_context_classify
        database = RCC_context_classify(split=['test'], transform=transforms, retname=True,
                                      do_classify='classify' in p.TASKS.NAMES,
                                      do_SSL=False,
                                    overfit=None)

    else:
        raise NotImplemented("test_db_name: Choose among PASCALContext and NYUD")

    return database


def get_val_dataloader(p, dataset):
    """ Return the validation dataloader """

    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=0)
    return testloader


""" 
    Loss functions 
"""

def get_loss_rudan(p, num_classes = None):
    """ Return loss function for a specific task """
    

    classify_loss = torch.nn.CrossEntropyLoss().to(device)


    return classify_loss

def get_criterion_rudan(p,task):
    """ Return training criterion for a given setup """
  
    from losses.loss_schemes import SingleTaskLoss

    if task == "classify":        
        loss_ft= get_loss_rudan(p)
        
    return SingleTaskLoss(loss_ft, task)
        

"""
    Optimizers and schedulers
"""
def get_optimizer(p, model):
    """ Return optimizer for a given model and setup """

    if p['model'] == 'cross_stitch': # Custom learning rate for cross-stitch
        print('Optimizer uses custom scheme for cross-stitch nets')
        cross_stitch_params = [param for name, param in model.named_parameters() if 'cross_stitch' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'cross_stitch' in name]
        assert(p['optimizer'] == 'sgd') # Adam seems to fail for cross-stitch nets
        optimizer = torch.optim.SGD([{'params': cross_stitch_params, 'lr': 100*p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                        momentum = p['optimizer_kwargs']['momentum'], 
                                        nesterov = p['optimizer_kwargs']['nesterov'],
                                        weight_decay = p['optimizer_kwargs']['weight_decay'])


    elif p['model'] == 'nddr_cnn': # Custom learning rate for nddr-cnn
        print('Optimizer uses custom scheme for nddr-cnn nets')
        nddr_params = [param for name, param in model.named_parameters() if 'nddr' in name]
        single_task_params = [param for name, param in model.named_parameters() if not 'nddr' in name]
        assert(p['optimizer'] == 'sgd') # Adam seems to fail for nddr-cnns 
        optimizer = torch.optim.SGD([{'params': nddr_params, 'lr': 100*p['optimizer_kwargs']['lr']},
                                     {'params': single_task_params, 'lr': p['optimizer_kwargs']['lr']}],
                                        momentum = p['optimizer_kwargs']['momentum'], 
                                        nesterov = p['optimizer_kwargs']['nesterov'],
                                        weight_decay = p['optimizer_kwargs']['weight_decay'])


    else: # Default. Same larning rate for all params
        print('Optimizer uses a single parameter group - (Default)')
        params = model.parameters()
    
        if p['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

        elif p['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
        
        else:
            raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer
   

def adjust_learning_rate(p, optimizer, epoch):
    """ Adjust the learning rate """

    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
