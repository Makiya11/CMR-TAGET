import csv
import os
import re

import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric
from transformers import BertTokenizer

from bert_score import BERTScorer
from rouge_score import rouge_scorer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

from model import get_git_model, load_partial_weights
from data_module import DataGenerator

class COCOCaptionMetrics(Metric):

    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            metrics=["bleu", "cider", "meteor", "rouge", "spice"],
            save=False,
            save_individual_scores=False,
            save_bootstrapped_scores=False,
            exp_dir=None,
    ):
        super().__init__()

        self.add_state("predictions", default=[])
        self.add_state("labels", default=[])
        self.add_state("ids", default=[])

        self.metrics = [metric.lower() for metric in metrics]
        self.save = save
        self.save_individual_scores = save_individual_scores
        self.save_bootstrapped_scores = save_bootstrapped_scores
        self.exp_dir = exp_dir

        self.num_metrics = 0
        if "bleu" in self.metrics:
            self.bleu = Bleu(4)
            self.num_metrics += 4
        if "meteor" in self.metrics:
            self.meteor = Meteor()
            self.num_metrics += 1
        if "rouge" in self.metrics:
            self.rouge = Rouge()
            self.num_metrics += 1
        if "cider" in self.metrics:
            self.cider = Cider()
            self.num_metrics += 1
        if "spice" in self.metrics:
            self.spice = Spice()
            self.num_metrics += 1

    def update(self, predictions, labels, ids):
        """
        Argument/s:
            predictions - the predicted captions must be in the following format:

                [
                    "a person on the snow practicing for a competition",
                    "group of people are on the side of a snowy field",
                ]

            labels - the corresponding labels must be in the following format:

                [
                    [
                        "Persons skating in the ice skating rink on the skateboard.",
                        "A snowboard sliding very gently across the snow in an enclosure.",
                        "A person on a snowboard getting ready for competition.",
                        "Man on snowboard riding under metal roofed enclosed area.",
                        "A snowboarder practicing his moves at a snow facility.",
                    ],
                    [
                        "There are mountains in the background and a lake in the middle.",
                        "a red fire hydrant in a field covered in snow",
                        "A fire hydrant in front of a snow covered field, a lake and
                        mountain backdrop.",
                        "A hydran in a snow covered field overlooking a lake.",
                        "An expanse of snow in the middle of dry plants",
                    ]
                ]

                or, if there is only one label per example (can still be in the above format):

                [
                    "Persons skating in the ice skating rink on the skateboard.",
                    "There are mountains in the background and a lake in the middle.",
                ]
            ids (list) - list of identifiers.
        """
        self.predictions.extend(list(predictions))
        self.labels.extend(list(labels))
        self.ids.extend(list(ids))

    def compute(self):
        """
        Compute the metrics from the COCO captioning task with and without DDP.

        Argument/s:
            stage - "val" or "test" stage of training.

        Returns:
            Dictionary containing the scores for each of the metrics
        """

        if torch.distributed.is_initialized():  # If DDP
            predictions_gathered = [None] * torch.distributed.get_world_size()
            labels_gathered = [None] * torch.distributed.get_world_size()
            ids_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(predictions_gathered, self.predictions)
            torch.distributed.all_gather_object(labels_gathered, self.labels)
            torch.distributed.all_gather_object(ids_gathered, self.ids)
            self.predictions = [j for i in predictions_gathered for j in i]
            self.labels = [j for i in labels_gathered for j in i]
            self.ids = [j for i in ids_gathered for j in i]

        return self.score()

    def score(self):

        predictions, labels = {}, {}
        for i, j, k in zip(self.ids, self.predictions, self.labels):
            predictions[i] = [re.sub(' +', ' ', j.replace(".", " ."))]
            labels[i] = [re.sub(' +', ' ', k.replace(".", " ."))]
        accumulated_scores = {}
        example_scores = {}
        if "bleu" in self.metrics:
            score, scores = self.bleu.compute_score(labels, predictions)
            accumulated_scores["chen_bleu_1"] = score[0]
            accumulated_scores["chen_bleu_2"] = score[1]
            accumulated_scores["chen_bleu_3"] = score[2]
            accumulated_scores["chen_bleu_4"] = score[3]
            example_scores["chen_bleu_1"] = scores[0]
            example_scores["chen_bleu_2"] = scores[1]
            example_scores["chen_bleu_3"] = scores[2]
            example_scores["chen_bleu_4"] = scores[3]
        if "meteor" in self.metrics:
            score, scores = self.meteor.compute_score(labels, predictions)
            accumulated_scores["chen_meteor"] = score
            example_scores["chen_meteor"] = scores
        if "rouge" in self.metrics:
            score, scores = self.rouge.compute_score(labels, predictions)
            accumulated_scores["chen_rouge"] = score
            example_scores["chen_rouge"] = scores
        if "cider" in self.metrics:
            score, scores = self.cider.compute_score(labels, predictions)
            accumulated_scores["chen_cider"] = score
            example_scores["chen_cider"] = scores
        if "spice" in self.metrics:
            score, scores = self.spice.compute_score(labels, predictions)
            accumulated_scores["chen_spice"] = score
            example_scores["chen_spice"] = scores
        accumulated_scores["chen_num_examples"] = len(predictions)

        if self.save:
            def save_reports():
                with open(os.path.join(self.exp_dir, "predictions.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["prediction", "label", "id"])
                    for row in zip(predictions.values(), labels.values(), self.ids):
                        writer.writerow([row[0][0], row[1][0], row[2]])

            if not torch.distributed.is_initialized():
                save_reports()
            elif torch.distributed.get_rank() == 0:
                save_reports()

        if self.save_individual_scores:
            def save_example_scores():
                df = pd.DataFrame(example_scores)
                df.to_csv(os.path.join(self.exp_dir, "individual_scores.csv"))
            if not torch.distributed.is_initialized():
                save_example_scores()
            elif torch.distributed.get_rank() == 0:
                save_example_scores()

        if self.save_bootstrapped_scores:
            df = pd.DataFrame(accumulated_scores, index=[0,])
            save_path = os.path.join(self.exp_dir, "bootstrapped_scores.csv")
            header = False if os.path.isfile(save_path) else True
            df.to_csv(save_path, mode="a", header=header, index=False)

        return accumulated_scores




def test_git_inference_single_image(param, weight_path, data_path):
    
    prefix = ''
    test_transforms =None
    inference = [1]
                
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)    
    dataset_test = DataGenerator(data_ver=param['data_ver'], file_path=data_path, phase='test', 
                                 num_frames=param['num_frames'],tokenizer=tokenizer)
    # model
    model = get_git_model(tokenizer, param)
    checkpoint = torch.load(weight_path, map_location='cuda:0')['state_dict']
    load_partial_weights(model, checkpoint)
    model.cuda()
    model.eval()

    # prefix
    prefix_encoding = tokenizer(prefix,
                                padding='do_not_pad',
                                truncation=True,
                                add_special_tokens=False,
                                max_length=param['max_text_len'])

    payload = prefix_encoding['input_ids']
    if len(payload) > param['max_text_len'] - 2:
        payload = payload[-(param['max_text_len'] - 2):]
    input_ids = [tokenizer.cls_token_id] + payload

    lst = []
    for data, idx in dataset_test:
        with torch.no_grad():
            result = model({
                'image': data['image'].unsqueeze(0).cuda(),
                'indication_tokens': data['indication_tokens'].unsqueeze(0).cuda(),
                'prefix': torch.tensor(input_ids).unsqueeze(0).cuda(),
                'inference': inference
            }, )
        cap = tokenizer.decode(result['predictions'][0].tolist(), skip_special_tokens=True)
        lst.append({'idx':idx, 'caption': dataset_test.df.iloc[idx]['caption'], 'output':cap})
        if output:
            print(f"caption :{dataset_test.df.iloc[idx]['caption']}")
            print(f"output':{cap}")
            print("-------------------------------------------------------------------------------------------------")

    df = pd.DataFrame(lst)
    return df

if __name__ == '__main__':
    output = True
    test_coco_metrics = COCOCaptionMetrics(metrics=["bleu", "cider", "meteor", "rouge"])
    hospital_area = 'OH_ind_v3'
    data_path = f'/data/aiiih/projects/nakashm2/multimodal/data/table/{hospital_area}.csv'
    save_path = 'output'
    weight_path = 'best_weight.ckpt'
    
    param = {'data_ver': 'sampling4', 
             'encoder':'SpaceTimeTransformer',
             'max_text_len': 100,
             'num_frames': 64
             }
    
    df = test_git_inference_single_image(param, weight_path, data_path)
    test_coco_metrics.update(df['output'], df['caption'], df['idx'])
    print(test_coco_metrics.score())
    df.to_csv(f'{save_path}/generated_{hospital_area}.csv', index=False)
        

