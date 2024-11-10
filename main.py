import argparse
from datetime import datetime
import pandas as pd
import torch
from data_layer.builder import collate_fn
from data_module import DataGenerator
from model import get_git_model
from traintest import TrainTest
from transformers import BertTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch

torch.manual_seed(0)
def main(params):
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(params)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)    

    dataset_train = DataGenerator(data_ver=params['data_ver'], file_path=params['data_path'], 
                                    phase='train', num_frames=params['num_frames'],
                                    tokenizer=tokenizer)
    dataset_val = DataGenerator(data_ver=params['data_ver'], file_path=params['data_path'], 
                                    phase='val', num_frames=params['num_frames'],
                                    tokenizer=tokenizer)
    
    dataset_test = DataGenerator(data_ver=params['data_ver'], file_path=params['data_path'], 
                                    phase='test', num_frames=params['num_frames'],
                                    tokenizer=tokenizer)
        
    
    ### Create the dataloader objects for each dataset
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params['batch_size'],
                                        shuffle=True, pin_memory=True,
                                        num_workers=params['num_workers'],
                                        collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=params['batch_size'],
                                        shuffle=False, pin_memory=True, 
                                        num_workers=params['num_workers'],
                                        collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=params['batch_size'],
                                        shuffle=False, pin_memory=True, 
                                        num_workers=params['num_workers'],
                                        collate_fn=collate_fn)
    
    model = get_git_model(tokenizer, params)
    ### move to GPU   
    model.to(device)
    
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable parameters: {n_train_params}')
    
    if params['half_precision']:
        model.half()
        ### optimizer
        grouped_parameters = [
            {"params": model.image_encoder.parameters(), 'lr': params['learning_rate']*0.1},
            {"params": model.textual.parameters(), 'lr': params['learning_rate']},
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, eps=1e-4)
        # optimizer = torch.optim.AdamW(model.parameters(), params['learning_rate'], eps=1e-4)
        
    else:
        grouped_parameters = [
            {"params": model.image_encoder.parameters(), 'lr': params['learning_rate']*0.1},
            {"params": model.textual.parameters(), 'lr': params['learning_rate']},
        ]
        optimizer = torch.optim.AdamW(grouped_parameters)
        # optimizer = torch.optim.SGD(grouped_parameters)
    
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)
    
    train_test = TrainTest(
        device=device,
        model=model,
        max_epochs=params['max_epochs'], optimizer=optimizer, scheduler=scheduler, 
        model_save_path=params['save_path'], tb_writer=params['tb_writer'], #sample_limit=20000,
        data_loader_train=train_loader, data_loader_val=val_loader,
        data_loader_test=test_loader)      
    train_test.fit()
    
    ### save info
    df_params = pd.DataFrame([params])
    df_params.to_csv(f"{params['save_path']}/info.csv")
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100, 
                        help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, 
                        help='learning rate')
    parser.add_argument('--num_frames', type=int, default=64, 
                        help='number of frames')
    parser.add_argument('--data_ver', type=str, default='sampling4', 
                        help='sampling')
    parser.add_argument('--encoder', type=str, default='SpaceTimeTransformer', 
                    help='video encoder')
    parser.add_argument('--max_text_len', type=int, default=100, 
                    help='max text length')
    parser.add_argument('--data_path', type=str, 
                        help='sampling')
    
    args = parser.parse_args()

    ### argument to dictionary
    params = vars(args)
    
    SAVE_DIR = 'output'
    
    run_name = str(datetime.timestamp(datetime.now())) + params['data_ver']
    add_params = {'num_workers': 8,
                  'run_name': run_name,
                  'tb_writer': None,
                  'data_path': params['data_path'],
                  'save_path': f"{SAVE_DIR}/results/{run_name}",
                  'half_precision': False}

    params.update(add_params)
    main(params)

