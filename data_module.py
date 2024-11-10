import os
import random
import sys
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import decord
from torchvision import transforms


class DataGenerator(Dataset):
    """Generates data for pytorch"""
    def __init__(self, data_ver, file_path, phase, tokenizer, num_frames, transform=None):
        """
        Initialization

        Input:
            data_ver: v1 no sentence separation, v2 sentence separation
            file_path: path to the csv file
            phase: train, test val phase (ex 'train') 
            transform: augmentation

        """
        self.data_ver = data_ver
        self.phase = phase
        self.file_path = file_path
        self.transform = transform 

        ### Video parameter
        self.num_frames = num_frames
        self.sample = 'uniform'
        self.fix_start = None
        
        ### Text parameter
        self.tokenizer = tokenizer
        
        self.trasform = init_transform_dict(**{})
        
        self.df = self.get_df()
       
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.df)

    def __getitem__(self, index):
        """Generate one batch of data"""
        ### Load text and video 
        ### Transform data dimmentions
        dic_vid = self.process_video(self.df.iloc[index]['video_path'])
        dic_txt = self.process_text(self.df.iloc[index]['indications_and_clinical_history'], self.df.iloc[index]['caption'])
        dic_vid.update(dic_txt)

        return dic_vid, index

    def process_video(self, video_path):
        frames = np.load(video_path)
        frames = np.repeat(frames[..., np.newaxis], 3, axis=-1)  
        frames = torch.from_numpy(frames)
        frames = frames.float() / 255
        frames = frames.permute(2, 3, 0, 1)
        # frames = frames.half()
        frames = self.trasform[self.phase](frames)
        return {'image': frames}

    def process_text(self, indication, target):
        max_text_len= 100
        prefix = ''
        prefix_encoding = self.tokenizer(
            prefix, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        indication_encoding = self.tokenizer(
            indication, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        target_encoding = self.tokenizer(
            target, padding='do_not_pad',
            add_special_tokens=False,
            truncation=True, max_length=max_text_len)
        
        need_predict = [0] * len(prefix_encoding['input_ids']) + [1] * len(target_encoding['input_ids'])
        payload = prefix_encoding['input_ids'] + target_encoding['input_ids']
        if len(payload) > max_text_len:
            payload = payload[:(max_text_len - 2)]
            need_predict = need_predict[:(max_text_len - 2)]
        input_ids = [self.tokenizer.cls_token_id] + payload + [self.tokenizer.sep_token_id]
        need_predict = [0] + need_predict + [1]
        
        # input_ids = indication_encoding['input_ids'] + input_ids
        # need_predict =  [0] * len(indication_encoding['input_ids']) + need_predict
        data = {
            'indication_tokens': torch.tensor(indication_encoding['input_ids']),
            'caption_tokens': torch.tensor(input_ids),
            'need_predict': torch.tensor(need_predict),
            'caption': {},
            'iteration': 0,
            'inference': 0
        }
        return data
    

    def get_df(self):

        df = pd.read_csv(self.file_path)
        df['indications_and_clinical_history'] = df['indications_and_clinical_history'].fillna('Clinical History')

        ### select phase
        df = df[df['train_test']==self.phase]
        df['video_path'] = '/'.join(self.file_path.split('/')[:7]) + f'/{self.data_ver}/' + df['AccessionNumber'].astype(str) + '.npy'
        df['text_path'] = df['caption']
        df = df.drop_duplicates(['AccessionNumber'])
        df = df.reset_index()
        print(f'{self.phase} data: {len(df)}')
        ## for debug
        return df



def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop, antialias=True),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res, antialias=True),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop, antialias=True),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res, antialias=True),
            normalize,
        ])
    }
    return tsfm_dict



if __name__ == "__main__":
    ### For debuging
    import sys
    import matplotlib.pyplot as plt
    from transformers import BertTokenizer

    file_path = '/data/aiiih/projects/nakashm2/multimodal/data/table/OH.csv'
    phase = 'test'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    transform = None
    data_ver= 'sampling1'
    x = DataGenerator(data_ver ,file_path, phase, tokenizer, 64, transform)
    for idx, i in enumerate(x):
        breakpoint()
        