import torch
from video_transformer import SpaceTimeTransformer
from layers.decoder import CaptioningModel
from layers.decoder import TransformerDecoderTextualHead
from search_algo import GeneratorWithBeamSearch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

def get_git_model(tokenizer, param):

    add_dim = True

    image_encoder = get_image_encoder(param)
    text_decoder = TransformerDecoderTextualHead(
        visual_feature_size=param.get('visual_feature_size', 768),
        vocab_size= 30522,
        hidden_size=768,
        num_layers=6,
        attention_heads=12,
        feedforward_size=768* 4,
        max_caption_length=128,
        mask_future_positions=True,
        padding_idx=0,
        decoder_type='bert_en',
        visual_projection_type='linearLn',
    )
    decoder = GeneratorWithBeamSearch(
        eos_index=tokenizer.sep_token_id,
        max_steps=param['max_text_len'],
        beam_size=1,
        length_penalty=0.6,
    )
    model = CaptioningModel(
        image_encoder,
        text_decoder,
        decoder=decoder,
        sos_index=tokenizer.cls_token_id,
        eos_index=tokenizer.sep_token_id,
        tokenizer=tokenizer,
        use_history_for_infer=True,
        loss_type='smooth',
        add_dim = add_dim,
        num_image_with_embedding=param.get('num_image_with_embedding')
    )
    return model

def get_image_encoder(param):
    weight_path = 'vid_encoder.pth'
    weight = torch.load(weight_path)
    model = SpaceTimeTransformer(num_frames=64,
                                time_init='zeros',
                                attention_style='frozen-in-time',)
    model.head = nn.Identity()
    model = load_partial_weights(model, weight)
            
    ### freeze layers
    for param in model.parameters():
        param.requires_grad = False        

    return model



def load_partial_weights(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(f'Number of layers in the model {len(model.state_dict())}')

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    print(f'Number of layers to load {len(pretrained_dict)}')
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model 


