import os
from datetime import datetime as dtt

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import tqdm

from utils import EarlyStopping, fcnSaveCheckpoint, fcnLoadCheckpoint, recursive_to_device


class TrainTest():
    def __init__(self,  device='gpu:0',
            model=None, max_epochs=5, criterion=None, optimizer=None, scheduler=None, 
            model_save_path=None, tb_writer=None, sample_limit=None, 
            data_loader_train=None, data_loader_val=None,
            data_loader_test=None):
        """
        Initialization
        Inputs:
        data_loader_train - data loader for the training set
        data_loader_val - data loader for the vaidation set
        data_loader_test - data loader for the test set
        model - model fitted
        criterion - loss function
        optimizer - optimizer
        scheduler - scheduler
        device - gpu or CPU
        params - parameters for the experiment
        tb_writer - tensorboard writer
        """

        super(TrainTest, self).__init__()

        ### do some checks
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        self.sample_limit = sample_limit
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.tb_writer = tb_writer
        self.model_save_path = model_save_path
        self.tqdm_disable = True
        self.metric_save_len = 20 ##disable
        self.softmax = nn.Softmax(dim=1)


    def run(self, phase, epoch):
        """Trains, validate or test model"""
        if phase == 'train':
            ##### switch to train
            self.model.train()
            dataloader = self.data_loader_train
        elif phase == 'val':
            ### switch to evaluate
            self.model.eval()
            dataloader = self.data_loader_val
        elif phase == 'test':
            ### switch to evaluate
            self.model.eval()
            dataloader = self.data_loader_test

        ### initialize loss for this epoch
        lossTotal = 0.0 #loss
        count = 0
        with torch.set_grad_enabled(phase=='train'):
            ### iterator over batches
            for data in tqdm.tqdm(dataloader,
                                        desc=f'Epoch {epoch}/{self.max_epochs}, {phase:>10}',
                                        total=len(dataloader),
                                        disable = self.tqdm_disable):
                # # Move the data to the GPU
                data = recursive_to_device(data, self.device)
                ### compute gradient and do SGD step
                self.optimizer.zero_grad()
                ###forward + backward + optimize
                loss_dict = self.model(data[0])
                loss = sum(loss_dict.values())
                loss = loss.mean()
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()
                            
                ### measure accuracy and record loss
                lossTotal += loss.item()
                
                count+=data[0]['image'].shape[0]
                if self.sample_limit:
                    if count > self.sample_limit:
                        break
                    
        mean_loss = lossTotal/len(dataloader)
        
        
        ### write tesorboard
        if self.tb_writer  is not None:
            self.tb_writer.add_scalar('Loss/{phase}', mean_loss, epoch)
                
        log = {f'{phase}_loss': mean_loss,
               f'{phase}_epoch': epoch}
        
        print(f'Phase: {phase}, Epoch: {epoch}, Loss: {mean_loss}')
        
        return log

    def fit(self, opt_dir='down'):
        """
        loop through epochs and fits model to data
        Inputs:
        opt_dir - 
        """
        ########################## training the model ##########################
        
        ### initialize model saving
        save_metric = EarlyStopping(patience=self.metric_save_len)

        print(f"start training model at {str(dtt.now().strftime('%H:%M:%S'))}")
        
        log_dic ={'train_log': [], 'val_log': []}
        ### train model and evaluate after every epoch
        for kEpoch in range(1, self.max_epochs+1):
            ### training
            train_log = self.run('train', kEpoch)

            ### evaluate
            val_log = self.run('val', kEpoch)
            
            log_dic['train_log'].append(train_log)
            log_dic['val_log'].append(val_log)
        
            self.scheduler.step(val_log['val_loss'])
        
            # flag_save_model, flag_early_stop = save_metric.update(train_log['train_loss'], kEpoch)
            flag_save_model, flag_early_stop = save_metric.update(val_log['val_loss'], kEpoch)
            
            fcnSaveCheckpoint(model=self.model, optimizer=self.optimizer,
                            epoch=kEpoch, metric_value=save_metric.best_score,
                            filepath=f'{self.model_save_path}/weights/{kEpoch}epoch_weight.ckpt')
            
            if flag_save_model and (self.model_save_path is not None):
                print(f'Save the weight to {self.model_save_path}/weights/best_weight.ckpt')
                ### train test
                fcnSaveCheckpoint(model=self.model, optimizer=self.optimizer,
                            epoch=kEpoch, metric_value=save_metric.best_score,
                            filepath=f'{self.model_save_path}/weights/best_weight.ckpt')
            if flag_early_stop:
                print(f"Stopped training early at epoch {kEpoch + 1}")
                break
        
        ### Save records
        save_record_dir = f'{self.model_save_path}'
        if not os.path.exists(save_record_dir):
            os.mkdir(save_record_dir)
        train_df = pd.DataFrame(log_dic['train_log'])
        val_df =pd.DataFrame(log_dic['val_log'])
        df = pd.concat([train_df, val_df], axis=1)
        df.to_csv(f"{save_record_dir}/train_log.csv", index=False)

        ### end of training
        print(f"end training model at {str(dtt.now().strftime('%H:%M:%S'))}")
        print('\n' + '#' * 10 + ' Beginning testing ' + '#' * 10)
        
        ########################## testing the model ###########################
        self.model, _, best_epoch, _ = fcnLoadCheckpoint(model=self.model,
                                                         optimizer=self.optimizer,
                                                         filepath=f'{self.model_save_path}/weights/best_weight.ckpt')
        test_log = self.run('test', kEpoch) #save_output
        train_log.update(test_log)
        # train_log.update(val_log)        
        return train_log