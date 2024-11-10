import os
import torch
import copy
import numpy as np
import glob
import pandas as pd

def fcnSaveCheckpoint(model, optimizer, epoch, metric_value, filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    ### saves checkpoint to allow resumption of training
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'best_metric': metric_value,}
    torch.save(state,filepath)
#end fcnSaveCheckpoint

def fcnLoadCheckpoint(model, optimizer, filepath):
    ### loads checkpoint to allow resumption of training
    epoch = 0
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("Previous metric: {:.3f}".format(checkpoint['best_metric']))
    else:
        print("Error no checkpoint found: {}".format(filepath))

    return model, optimizer, epoch, checkpoint['best_metric']
#end fcnLoadCheckpoint

def recursive_to_device(d, device, **kwargs):
    if isinstance(d, tuple) or isinstance(d, list):
        return [recursive_to_device(x, device, **kwargs) for x in d]
    elif isinstance(d, dict):
        return dict((k, recursive_to_device(v, device)) for k, v in d.items())
    elif isinstance(d, torch.Tensor) or hasattr(d, 'to'):
        #return d.to(device, non_blocking=True)
        return d.to(device, **kwargs)
    else:
        return d


class EarlyStopping():
    def __init__(self, patience=10, delta=0, min_epoch=15):        
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.min_epoch = min_epoch
        self.flag_save_model = False
        self.flag_early_stop = False
    
    def update(self, x, epoch):
        if self.best_score is None:
            self.best_score = x
            self.flag_save_model = True
        elif x > self.best_score + self.delta:
            self.counter += 1
            self.flag_save_model = False
            print(f'Early Stopping: {self.counter} / {self.patience}')
            if self.counter >= self.patience and epoch> self.min_epoch:
                self.flag_early_stop = True
        else:
            self.best_score = x
            self.flag_save_model = True
            self.counter = 0
        return self.flag_save_model, self.flag_early_stop
