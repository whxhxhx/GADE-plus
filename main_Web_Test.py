import logging
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import mean
import argparse
from utils.dataset_generation import *
from torch.utils.data import DataLoader
from logger import set_logger
from utils.utils import *
from pytorch_transformers import AdamW, WarmupLinearSchedule

from GADE_framework.AnchorLRM import LRM
from GADE_framework.AnchorLRMVariant import LRM_Albert, LRM_Roberta
from GADE_framework.AnchorGNN import AnchorGNN


os.environ["CUDA_VISIBLE_DEVICE"] = "0, 1"

f1_list = []



def aux_ce_loss(loss_func, anchor_nums, anchor_logits):
    proxy_labels = [0] * (anchor_nums // 2) + [1] * (anchor_nums // 2)
    proxy_labels = torch.Tensor(proxy_labels).to("cuda:{:d}".format(args.gpu[0])).long()
    aux_loss = loss_func(anchor_logits, proxy_labels)
    return aux_loss


### comp
def anchor_reg_loss(x_emb, labels, anchors):
    proxy_labels = torch.arange(0, 2).to("cuda:{:d}".format(args.gpu[0]))
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, proxy_labels.T).float().to("cuda:{:d}".format(args.gpu[0])) #bz, cls
    n, d = x_emb.shape
    a_n = anchors.size(0)
    
    neg_centroid = torch.mean(anchors[:a_n//2,:], dim=0)
    pos_centroid = torch.mean(anchors[a_n//2:,:], dim=0)
    anchors_ = torch.stack((neg_centroid, pos_centroid), dim=0)

    x_emb_norm = F.normalize(x_emb, p=2, dim=-1)
    anchors_norm = F.normalize(anchors_, p=2, dim=-1)

    # feat_dot_prototype = torch.matmul(x_emb, anchors_.transpose(-1,-2))
    feat_dot_prototype = torch.matmul(x_emb_norm, anchors_norm.transpose(-1,-2)) / args.tau
    # feat_dot_prototype = torch.sum((x_emb_norm.unsqueeze(1)-anchors_norm.unsqueeze(0))**2, dim=-1)
    
    logits_max, _ = torch.max(feat_dot_prototype, dim=1, keepdim=True)
    logits = feat_dot_prototype - logits_max.detach()

    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(1) 

    # loss
    mse_loss = -1.0 * mean_log_prob_pos.mean()

    return mse_loss


def test_model(iter, logger, gim, lrm, criterion, test_step=None, prefix='Test'):
    gim.eval()
    lrm.eval()

    scores = []
    labels = []

    for j, batch in enumerate(iter):
        with torch.no_grad():
            feature, label, masks = lrm(batch)
            masks = masks.view(-1)
            label = label.view(-1)[masks == 1].long()

            if args.aux_ce_reg and args.p2a_regularization:
                pred, x_emb, anchor_vecs, anchor_logits = gim(feature)
            elif args.aux_ce_reg and not args.p2a_regularization:
                pred, anchor_logits = gim(feature)
            elif not args.aux_ce_reg and args.p2a_regularization:
                pred, x_emb, anchor_vecs = gim(feature)
            else:
                pred = gim(feature)

            pred = pred[masks == 1]
            loss = criterion(pred, label)
            pred = F.softmax(pred, dim=1)
            p, r, acc = accuracy(pred, label)
            logger.info(
                '{}\t[{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(prefix, j + 1,
                                                                                                       len(iter), loss,
                                                                                                       acc,
                                                                                                       p, r))
            assert pred.shape[0] == label.shape[0]
            scores += pred[:, 1].detach().cpu().numpy().tolist()
            labels += label.detach().cpu().numpy().tolist()

    p, r, f1, acc = calculate_f1(scores, labels)
    logger.info(
        '{}\tPrecison {:.3f}\tRecall {:.3f}\tF1-score {:.3f}\tAccuracy {:.3f}'.format(prefix, p, r, f1, acc))

    return f1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--seed', default=28, type=int)
    parser.add_argument('--exp_dir', default=".", type=str)
    parser.add_argument('--log_freq', default=5, type=int)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_node', type=int, default=165)

    # Optimization args
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--embed_lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float,default=0.4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)

    # Data path args
    parser.add_argument('--checkpoint_path', default="./saved_ckpt/checkpoints", type=str)
    parser.add_argument('--data_type', type=str, default='Web_Test')
    parser.add_argument('--model_name', default='GADE_300', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+')
    parser.add_argument('--gcn_layer', default=2, type=int)

    parser.add_argument('--anchor_nums', default=20, type=int)
    parser.add_argument('--num_pers', default=4, type=int)
    parser.add_argument('--p2a_coff', type=float, default=0.1)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--p2a_regularization', action='store_true', default=False)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--aux_ce_reg', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lrm_type', default="bert", type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gcn_dim = 768
    kfold = args.kfold

    params = args.__dict__

    args.entity_path = 'datasets/' + args.data_type + '/target_entities.txt'
    args.data_path = 'datasets/' + args.data_type + '/TDD_dataset.json'
    args.description_path = 'datasets/' + args.data_type + '/entity_desc.json'
    ent_list = load_entity_list(args.entity_path)

    logger = set_logger(os.path.join("./saved_ckpt/checkpoints/Web_Test_results", args.data_type +"_" + args.model_name + ".log"))

    for i in range(kfold):
        logger.info("The {}-th fold testing begins!".format(i))
        if args.lrm_type == "bert":
            lrm = LRM(hid_dim=args.hid_dim, anchor_nums=args.anchor_nums, num_pers=args.num_pers, max_seq_length=args.max_seq_length, device=args.gpu)
        elif args.lrm_type == "roberta":
            lrm = LRM_Roberta(hid_dim=args.hid_dim, anchor_nums=args.anchor_nums, num_pers=args.num_pers, max_seq_length=args.max_seq_length, device=args.gpu)
        elif args.lrm_type == "albert":
            lrm = LRM_Albert(hid_dim=args.hid_dim, anchor_nums=args.anchor_nums, num_pers=args.num_pers, max_seq_length=args.max_seq_length, device=args.gpu)

        gim = AnchorGNN(gcn_dim, args.hid_dim, 2, args.gcn_layer, args.dropout, anchor_nums=args.anchor_nums, num_pers=args.num_pers, device=args.gpu, aux_ce=args.aux_ce_reg, p2a_reg=args.p2a_regularization)
        tokenizer = lrm.tokenizer
        input_tokens, label_inputs, desc_tokens = generate_data(ent_list, args.description_path,
                                                                args.data_path, tokenizer,
                                                                args.max_seq_length)     

        test_ent = yield_example(ent_list, input_tokens, label_inputs, desc_tokens)
        test_dataset = ComparisonDataset(test_ent)
        test_iter = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

        checkpoint_path = args.checkpoint_path + '/' + args.model_name
        start_epoch = 0
        start_f1 = 0.0

        ckp_path = checkpoint_path + "/{}_best.pth".format(i)
        checkpoint = torch.load(ckp_path)
        lrm.load_state_dict(checkpoint["lrm"], strict=False)
        gim.load_state_dict(checkpoint["gim"], strict=False)

        lrm = lrm.to(lrm.device)
        gim = gim.to(lrm.device)
        criterion = nn.CrossEntropyLoss().to(lrm.device)

        if args.aux_ce_reg:
            aux_criter = nn.CrossEntropyLoss().to(lrm.device)
        else:
            aux_criter = None
        
        logger.info("load from {}".format(ckp_path))
        f1_score = test_model(iter=test_iter, logger=logger, gim=gim, lrm=lrm, prefix='Test', criterion=criterion, test_step=i + 1)
        f1_list.append(f1_score)
    logger.info("5 fold test f1-scores is {}".format(f1_list))
    logger.info("The average f1 score of 5 fold cross validation is {}".format(mean(f1_list)))

    
