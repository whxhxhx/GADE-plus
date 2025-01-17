import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel, DistilBertTokenizer, DistilBertConfig, DistilBertModel, RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AlbertTokenizer, AlbertConfig, AlbertModel

os.environ["CUDA_VISIBLE_DEVICE"] = "0, 1"
import numpy as np
from torch.distributions.normal import Normal
INF = 1e20
VERY_SMALL_NUMBER = 1e-12



class LRM(nn.Module):
    def __init__(self, hid_dim=256, anchor_nums=20, num_pers=4, max_seq_length=128, device=0):
        super(LRM, self).__init__()
        if not isinstance(device, list):
            device = [device]
        self.device = torch.device("cuda:{:d}".format(device[0]))
        self.max_seq_length = max_seq_length

        self.encoder_pretrain_path = '/home/ubuntu/DM_Group/Wen/GADE_plus/plms/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.encoder_pretrain_path, do_lower_case=True)
        self.config = BertConfig.from_pretrained(self.encoder_pretrain_path)

        if torch.cuda.is_available() and len(device) > 1:
            self.model = nn.DataParallel(BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config), device_ids=device)
        else:
            self.model = BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config)

        for param in self.model.parameters():
            param.requires_grad = True

        self.dim = 768
        self.hidden_dim = hid_dim
        self.anchor_nums = anchor_nums
        self.num_pers = num_pers

    def encode_feature(self, cand_docs):
        input_ids = []
        segment_ids = []
        input_masks = []

        for s in cand_docs["input_tokens"]:
            tokens = ["[CLS]"] + s + ["[SEP]"] + cand_docs["description_token"] + ["[SEP]"]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            seg_pos = len(s) + 2
            seg_ids = [0] * seg_pos + [1] * (len(tokens) - seg_pos)
            mask = [1] * len(tokens)
            padding = [0] * (self.max_seq_length - len(tokens))
            tokens += padding
            input_ids.append(tokens)
            seg_ids += padding
            segment_ids.append(seg_ids)
            mask += padding
            input_masks.append(mask)

        input_ids = torch.LongTensor(input_ids).to(self.device)
        segment_ids = torch.LongTensor(segment_ids).to(self.device)
        input_masks = torch.LongTensor(input_masks).to(self.device)
        outputs, pooled_output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks)
        
        return pooled_output


    def single_forward(self, cand_docs, max_n):
        features = self.encode_feature(cand_docs)

        num_nodes, fdim = features.shape

        labels = cand_docs["labels"].copy()
        mask = [1] * len(cand_docs["labels"])

        labels += [-10] * (max_n - num_nodes)
        mask += [0] * (max_n - num_nodes)
        features = torch.cat([features, torch.zeros((max_n - num_nodes, fdim), dtype=torch.float32).to(self.device)], dim=0)

        return features, labels, mask

    def forward(self, batch_data):
        features = []
        inter_strength_mat = []
        label = []
        mask = []
        anchor_vecs = []

        max_n = 0

        for bd in batch_data:
            if len(bd["labels"]) > max_n:
                max_n = len(bd["labels"])

        for bd in batch_data:
            feat, l, m = self.single_forward(bd, max_n)
            features.append(feat)
            label.append(l)
            mask.append(m)

        features = torch.stack(tuple(features), dim=0).to(self.device)
        label = torch.Tensor(label).to(self.device)
        mask = torch.Tensor(mask).to(self.device)
       
        return features, label, mask
   