
"""
Implementations of GIT in PyTorch

A Generative Image-to-text Transformer for Vision and Language
https://arxiv.org/pdf/2205.14100.pdf
Acknowledgments:
    https://github.com/microsoft/GenerativeImage2Text/blob/main/generativeimage2text
"""


import warnings
from torch.nn import functional as F
import torch
import logging
from torch import nn
from pprint import pformat
import functools

def create_projecton_layer(visual_projection_type,
                           visual_feature_size,
                           textual_feature_size,
                           ):
    if visual_projection_type is None:
        visual_projection = nn.Linear(
            visual_feature_size, textual_feature_size
        )
    elif visual_projection_type == 'linearLn':
        visual_projection = nn.Sequential(
            nn.Linear(
                visual_feature_size, textual_feature_size
            ),
            nn.LayerNorm(textual_feature_size),
        )
    else:
        raise NotImplementedError(visual_projection_type)
    return visual_projection

class WordAndPositionalEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        max_caption_length: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        #self.padding_idx = padding_idx

        #self.words = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.words = nn.Embedding(vocab_size, hidden_size)

        # We provide no "padding index" for positional embeddings. We zero out
        # the positional embeddings of padded positions as a post-processing.
        self.positions = nn.Embedding(max_caption_length, hidden_size)
        self.layer_norm = nn.LayerNorm(
            hidden_size, eps=1e-8, elementwise_affine=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens):
        position_indices = self._create_position_indices(tokens)
        # shape: (batch_size, max_caption_length, hidden_size)
        word_embeddings = self.words(tokens)
        position_embeddings = self.positions(position_indices)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = self.layer_norm(word_embeddings + position_embeddings)
        embeddings = self.dropout(embeddings)

        #token_mask = (tokens != self.padding_idx).unsqueeze(-1)
        #embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings

    @functools.lru_cache(maxsize=128)
    def _create_position_indices(self, tokens):

        # Create position indices of the same size as token indices.
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(
            max_caption_length, dtype=tokens.dtype, device=tokens.device
        )
        # shape: (batch_size, max_caption_length)
        positions = positions.unsqueeze(0).expand(batch_size, max_caption_length)
        return positions

class BertEncoderAsDecoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, tgt, memory,
                tgt_mask=None,
                #memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                tgt_bi_valid_mask=None,
                encoder_history_states=None,
                # tgt_bi_valid_mask: N x num_tgt
                ):
        assert tgt_key_padding_mask is None, 'not supported'
        assert tgt_mask.dim() == 2
        assert tgt_mask.shape[0] == tgt_mask.shape[1]
        # tgt_mask should always be 0/negative infinity
        # mask
        tgt = tgt.transpose(0, 1)
        memory = memory.transpose(0, 1)
        hidden_states = torch.cat((memory, tgt), dim=1)
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        top_right = torch.full((num_memory, num_tgt), float('-inf'), device=tgt.device, dtype=dtype,)
        bottom_left = torch.zeros((num_tgt, num_memory), dtype=dtype, device=tgt_mask.device,)
        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full((memory.shape[0], memory.shape[1]), fill_value=False, device=device)
        # if it is False, it means valid. That is, it is not a padding
        assert memory_key_padding_mask.dtype == torch.bool
        zero_negative_infinity = torch.zeros_like(memory_key_padding_mask, dtype=tgt.dtype)
        zero_negative_infinity[memory_key_padding_mask] = float('-inf')
        full_attention_mask = full_attention_mask.expand((memory_key_padding_mask.shape[0], num_memory + num_tgt, num_memory + num_tgt))
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        if tgt_bi_valid_mask is not None:
            # verify the correctness
            bs = full_attention_mask.shape[0]
            # during inference, tgt_bi_valid_mask's length is not changed, but
            # num_tgt can be increased
            max_valid_target = tgt_bi_valid_mask.shape[1]
            mask = tgt_bi_valid_mask[:, None, :].expand((bs, num_memory+num_tgt, max_valid_target))
            full_attention_mask[:, :, num_memory:(num_memory+max_valid_target)][mask] = 0

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]
        
        if encoder_history_states is None:
            result = self.encoder(
                hidden_states=hidden_states,
                attention_mask=full_attention_mask,
                encoder_history_states=encoder_history_states,
            )
            result = list(result)
            result[0] = result[0][:, num_memory:].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result[0], result[1]
            else:
                # make it back-compatible
                return result[0]
        else:
            encoder_out = self.encoder(
                hidden_states=hidden_states[:, -1:],
                attention_mask=full_attention_mask[:, :, -1:],
                encoder_history_states=encoder_history_states,
            )
            result = encoder_out[0].transpose(0, 1)
            if self.encoder.output_hidden_states:
                return result, encoder_out[1]
            else:
                return result

def create_decoder(decoder_type, norm_type,
                   textual_feature_size,
                   attention_heads,
                   feedforward_size,
                   dropout,
                   num_layers,
                   output_hidden_states=False,
                   use_mlp_wrapper=None,
                   ):
    assert norm_type in ['post', 'pre']
    if decoder_type is None:
        assert NotImplemented
    elif decoder_type == 'bert_en':
        from .bert import BertConfig
        from .bert.modeling_bert import BertEncoder
        config = BertConfig(
            vocab_size_or_config_json_file=30522,
            hidden_size=textual_feature_size,
            num_hidden_layers=num_layers,
            num_attention_heads=attention_heads,
            intermediate_size=feedforward_size,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
        )
        config.pre_norm=(norm_type == 'pre')
        config.use_mlp_wrapper = use_mlp_wrapper
        config.output_hidden_states = output_hidden_states
        encoder = BertEncoder(config)
        return BertEncoderAsDecoder(encoder)

class TransformerDecoderTextualHead(nn.Module):
    # used by unifusiondecoder and imageencodertextdecoder pipelines
    def __init__(
        self,
        visual_feature_size: int,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        attention_heads: int,
        feedforward_size: int,
        dropout: float = 0.1,
        norm_type: str = "post",
        mask_future_positions: bool = True,
        max_caption_length: int = 30,
        max_indication_length: int = 0,
        padding_idx: int = 0,
        decoder_type=None,
        visual_projection_type=None,
        not_tie_weight=None,
        output_hidden_states=None,
        use_mlp_wrapper=None,
        cosine_linear=False,
    ):
        super().__init__()
        self.visual_feature_size = visual_feature_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.max_indication_length = max_indication_length
        assert mask_future_positions
        self.padding_idx = padding_idx

        if visual_feature_size is not None:
            self.visual_projection = create_projecton_layer(
                visual_projection_type, visual_feature_size, self.textual_feature_size)
        else:
            self.visual_projection = nn.Identity()
        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.textual_feature_size,
            dropout=dropout,
            max_caption_length=max_caption_length,
            padding_idx=padding_idx,
        )
        
        self.embedding_ind = WordAndPositionalEmbedding(
            self.vocab_size,
            self.textual_feature_size,
            dropout=dropout,
            max_caption_length=max_caption_length,
            padding_idx=padding_idx,
        )
            
        self.transformer = create_decoder(
            decoder_type=decoder_type,
            norm_type=norm_type,
            textual_feature_size=self.textual_feature_size,
            attention_heads=self.attention_heads,
            feedforward_size=self.feedforward_size,
            dropout=dropout,
            num_layers=self.num_layers,
            output_hidden_states=output_hidden_states,
            use_mlp_wrapper=use_mlp_wrapper,
        )
        self.apply(self._init_weights)

        if cosine_linear:
            assert NotImplementedError
        else:
            # Create an output linear layer and tie the input and output word
            # embeddings to reduce parametejs.
            self.output = nn.Linear(self.textual_feature_size, vocab_size)
        if not not_tie_weight:
            self.output.weight = self.embedding.words.weight

    @property
    def textual_feature_size(self):
        return self.hidden_size

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            #if module.padding_idx is not None:
                #module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        hidden_states,
        indication_tokens,
        caption_tokens,
        hidden_valid_mask=None, # can be None
        caption_lengths=None, # useless
        bi_valid_mask_caption=None,
        #caption_mask=None,
        encoder_history_states=None,
        return_dict=False,
    ):
        if return_dict:
            ret = {}

        projected_visual_features = self.visual_projection(hidden_states) if hidden_states is not None else None
        if return_dict:
            ret['projected_visual_features'] = projected_visual_features
        # batch_size, max_indication_length = indication_tokens.size()
        batch_size, max_caption_length = caption_tokens.size()
        
        indication_embeddings = self.embedding_ind(indication_tokens)
        caption_embeddings = self.embedding(caption_tokens)
        # if projected_visual_features.shape[0] > 1:
        #     indication_embeddings = indication_embeddings.repeat(3, 1, 1)
        
        projected_visual_features = torch.cat([projected_visual_features, indication_embeddings], dim=1)

        # An additive mask for masking the future (one direction).
        uni_mask_zero_neg = self._generate_future_mask(
            max_caption_length, caption_embeddings.dtype, caption_embeddings.device
        )

        # We transpose the first two dimensions of tokens embeddings and visual
        # features, as required by decoder.
        caption_embeddings = caption_embeddings.transpose(0, 1)
        
        if projected_visual_features is not None:
            projected_visual_features = projected_visual_features.transpose(0, 1)
        else:
            projected_visual_features = torch.zeros(
                (0, caption_embeddings.shape[1], caption_embeddings.shape[2]),
                dtype=caption_embeddings.dtype,
                device=caption_embeddings.device,
            )
            
        extra_param = {}
        if bi_valid_mask_caption is not None:
            extra_param = {'tgt_bi_valid_mask': bi_valid_mask_caption}
        if not isinstance(self.transformer, torch.nn.modules.transformer.TransformerDecoder):
            extra_param['encoder_history_states'] = encoder_history_states
        # if transformer here is the pytorch/decoder, there is no chance, the
        # output is always tensor
        trans_out = self.transformer(
            caption_embeddings,
            projected_visual_features,
            memory_key_padding_mask=(hidden_valid_mask.logical_not() if hidden_valid_mask is not None else None),
            tgt_mask=uni_mask_zero_neg,
            #tgt_key_padding_mask=caption_mask,
            #encoder_history_states=encoder_history_states,
            **extra_param,
        )
        if isinstance(trans_out, tuple):
            textual_features = trans_out[0]
        else:
            assert isinstance(trans_out, torch.Tensor)
            textual_features = trans_out
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)
        if return_dict:
            ret['textual_features'] = textual_features

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)
        if isinstance(trans_out, tuple):
            if return_dict:
                ret['output_logits'] = output_logits
                ret['history'] = trans_out[1]
                return ret
            else:
                return output_logits, trans_out[1]
        else:
            if return_dict:
                ret['output_logits'] = output_logits
                return ret
            else:
                return output_logits

    def _generate_future_mask(
        self, size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

def convert2valid(shape, length=None, device='cuda'):
    if length is None:
        valid = torch.full(shape, fill_value=True, device=device)
    else:
        ones = torch.ones(shape, device=device)
        valid = ones.cumsum(dim=1) <= length.unsqueeze(1)
    return valid

class SmoothLabelCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1, log_prefix='', ignore_index=None):
        super().__init__()
        self.eps = eps
        self.log_soft = nn.LogSoftmax(dim=1)
        #self.kl = nn.KLDivLoss(reduction='batchmean')
        self.kl = nn.KLDivLoss(reduction='none')

        # for verbose printing only
        #self.register_buffer('iter', torch.tensor(0))
        self.iter = 0
        self.max_loss = 0
        self.min_loss = 0
        self.log_prefix = log_prefix
        self.ignore_index = ignore_index

    def forward(self, feature, target):
        # if it is fp16, convert it to fp32 explicitly as some trainer will not
        # do automatically
        feature = feature.float()
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target[valid_mask]
            feature = feature[valid_mask]
        assert target.numel() > 0
        debug_print = (self.iter % 100) == 0
        self.iter += 1
        eps = self.eps
        n_class = feature.size(1)
        one_hot = torch.zeros_like(feature).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(feature)
        if debug_print:
            with torch.no_grad():
                prob = torch.nn.functional.softmax(feature.detach(), dim=1)
                num = feature.size(0)
                avg_prob = prob[torch.arange(num), target].mean()
                logging.info('{}: iter={}, avg pos = {}, max loss = {}, min loss = {}'.format(
                    self.log_prefix,
                    self.iter,
                    avg_prob,
                    self.max_loss,
                    self.min_loss,
                ))
                self.max_loss = 0
                self.min_loss = 10000000
        loss = self.kl(log_prb, one_hot)
        with torch.no_grad():
            if len(loss) > 0:
                self.max_loss = max(self.max_loss, loss.max().cpu())
                self.min_loss = min(self.min_loss, loss.min().cpu())
        return loss.sum(dim=1).mean()


class CaptioningModel(nn.Module):
    def __init__(
        self,
        visual,
        textual,
        sos_index=1,
        eos_index=2,
        decoder=None,
        loss_type=None,
        context_not_share_embedding=False,
        scst=False,
        tokenizer=None,
        scst_temperature=1.,
        use_history_for_infer=False,
        pooling_images=None,
        num_image_with_embedding=0,
        add_dim= None,
    ):
        super().__init__()
        self.image_encoder = visual
        self.textual = textual
        self.padding_idx = self.textual.padding_idx

        # These boundary indices are needed for beam search.
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.decoder = decoder
        self.scst = scst
        self.tokenizer = tokenizer
        self.add_dim =  add_dim

        if self.scst:
            raise NotImplementedError

        if loss_type is None:
            self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        elif loss_type == 'smooth':
            self.loss = SmoothLabelCrossEntropyLoss(ignore_index=self.padding_idx)
        else:
            raise NotImplementedError(loss_type)
        #self.use_masked_as_input_for_train = use_masked_as_input_for_train

        self.verbose = {'num_has_image': 0, 'num_no_image': 0}
        self.context_not_share_embedding = context_not_share_embedding
        if context_not_share_embedding:
            self.context_embedding = self.textual.embedding.clone()
            # check whether the parameters are shared or not. it should not
            # share
        self.use_history_for_infer = use_history_for_infer
        self.pooling_images = pooling_images

        if num_image_with_embedding:
            logging.info('creating temperal embedding')
            self.img_temperal_embedding = nn.ParameterList(
                nn.Parameter(torch.zeros(1, 1, self.textual.visual_feature_size)) for _ in range(num_image_with_embedding)
            )
        self.num_image_with_embedding = num_image_with_embedding

    def forward(self, batch):
        result = self.forward_one(batch, return_info=False)
        return result

        # shape: (batch_size, channels, height, width)
    def forward_one(self, batch, return_info=False):
        # shape: (batch_size, max_caption_length, vocab_size)
        if 'image' in batch:
            if isinstance(batch['image'], (list, tuple)):
                features = [self.image_encoder(im) for im in batch['image']]
                
                if self.num_image_with_embedding:
                    features = [f + e for f, e in zip(features, self.img_temperal_embedding)]
                if self.pooling_images is None:
                    visual_features = torch.cat(features, dim=1)
                elif self.pooling_images == 'avg':
                    visual_features = torch.stack(features, dim=1).mean(dim=1)
                else:
                    raise NotImplementedError
            else:
                visual_features = self.image_encoder(batch['image'])
                if self.add_dim:
                    visual_features = visual_features.unsqueeze(1)
        else:
            visual_features = None
        visual_features_valid = None
        if 'context' in batch:
            context_embedding = self.context_embedding if self.context_not_share_embedding else self.textual.embedding
            all_context = [visual_features]
            all_valid = [convert2valid(visual_features.shape[:2])]
            for info in batch['context']:
                context = context_embedding(info['tokens'])
                valid = convert2valid(info['tokens'].shape, info['length'])
                all_context.append(context)
                all_valid.append(valid)
            visual_features = torch.cat(all_context, dim=1)
            visual_features_valid = torch.cat(all_valid, dim=1)
        
        # if not self.training or (not self.scst):
        return self.forward_one_ce(batch, visual_features, visual_features_valid, return_info)


    def forward_one_ce(self, batch, visual_features, visual_features_valid, return_info):
    
        has_image = (visual_features is not None)
        assert has_image == ('image' in batch)
        if not batch['inference'][0]:
            indication_token_input = batch["indication_tokens"]
            caption_token_input = batch["caption_tokens"]
            #caption_lengths = batch["caption_lengths"]
            
            output_logits = self.textual(
                visual_features,
                indication_token_input,
                caption_token_input,
                #caption_lengths=caption_lengths,
                hidden_valid_mask=visual_features_valid,
                bi_valid_mask_caption=batch.get('bi_valid_mask_caption'),
            )
            output_dict = {}
            #output_logits = x['output_logits']
            #ipdb> output_logits.shape
            #torch.Size([2, 13, 30522])
            #ipdb> batch['caption_tokens'].shape
            #torch.Size([2, 13])
            if 'need_predict' in batch:
                target = batch["caption_tokens"].clone()
                if self.padding_idx is not None:
                    target[batch['need_predict'] == 0] = self.padding_idx
            else:
                assert ValueError()
                #target = batch["caption_tokens"]
            need_predict = batch['need_predict']
            feat = output_logits[:, :-1].contiguous()
            target = target[:, 1:].contiguous()
            need_predict = need_predict[:, 1:].contiguous()
            feat = feat.view(-1, self.textual.vocab_size)
            target = target.view(-1)
            need_predict = need_predict.view(-1)

            valid_mask = need_predict == 1
            #valid_mask2 = target != self.padding_idx
            #assert (valid_mask.long() - valid_mask2.long()).abs().sum().cpu() == 0
            target = target[valid_mask]
            feat = feat[valid_mask]
            loss = self.loss(feat, target)
            if (self.verbose['num_has_image'] + self.verbose['num_no_image']) % 200 == 0:
                logging.info(self.verbose)
            hint = 'l' if 'context_target_type' not in batch else batch['context_target_type'][0]
            if has_image:
                output_dict.update({'vl_{}_loss'.format(hint): loss})
                self.verbose['num_has_image'] += 1
            else:
                output_dict.update({'l_{}_loss'.format(hint): loss})
                self.verbose['num_no_image'] += 1

            if return_info:
                output_dict['feat'] = feat
        else:
            indication_token_input = batch["indication_tokens"]
            output_dict = self.infer(batch, visual_features, visual_features_valid)
        return output_dict

    def infer(self, batch, visual_features, visual_features_valid,
              search_param=None):
        
        indication_token_input = batch["indication_tokens"]
        
        batch_size = visual_features.size(0)
        if 'prefix' not in batch:
            start_predictions = visual_features.new_full(
                (batch_size,1), self.sos_index
            ).long()
        else:
            # if batch size is larger than 1, the prefix length could be
            # different, and we have to padding non-valid data, which
            # is not supported
            assert len(batch['prefix']) == 1, 'not supported'
            start_predictions = batch['prefix'].long()

        self.prev_encoded_layers = None
        # Add image features as a default argument to match callable
        # signature accepted by beam search class (partial captions only).
        decoding_step = functools.partial(
            self.decoding_step, visual_features, indication_token_input, visual_features_valid,
            batch.get('bi_valid_mask_caption')
        )

        search_param = search_param or {}
        # the start_predictions are not in predicted_caption
        predicted_caption, logprobs = self.decoder.search(
            start_predictions, decoding_step, **search_param
        )
        if 'prefix' in batch:
            # we need to remove prefix from predicted_caption
            predicted_caption = predicted_caption[:, start_predictions.shape[1]:]
        output_dict = {
            'predictions': predicted_caption,
            'logprobs': logprobs,
        }
        return output_dict

    def decoding_step(
        self, visual_features, indication_token_input, visual_features_valid, bi_valid_mask_caption, partial_captions
    ):
        # Expand and repeat image features while doing beam search.
        batch_size = visual_features.shape[0]
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            batch_size, num_token, channels = visual_features.size()
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, num_token, channels
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a timestep. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        logits = self.textual(
            visual_features,
            indication_token_input,
            partial_captions,
            caption_lengths=caption_lengths,
            hidden_valid_mask=visual_features_valid,
            bi_valid_mask_caption=bi_valid_mask_caption,
            encoder_history_states=self.prev_encoded_layers,
        )
        if self.scst or self.use_history_for_infer:
            if isinstance(logits, tuple) and len(logits) == 2:
                if self.prev_encoded_layers is None:
                    self.prev_encoded_layers = logits[1]
                else:
                    self.prev_encoded_layers = [torch.cat((p, c), dim=1) for p, c in
                                                zip(self.prev_encoded_layers, logits[1])]
                #self.prev_encoded_layers = None
                logits = logits[0]
        return logits[:, -1, :].float()

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


