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

    feat_dot_prototype = torch.matmul(x_emb_norm, anchors_norm.transpose(-1,-2)) / args.tau

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


def train_model(iter, checkpoint_path, logger, fold, gim, lrm, optimizer, criterion, aux_criter, epoch_num,
          start_epoch=0, scheduler=None, test_iter=None, val_iter=None, log_freq=1, start_f1=None):
    
    step = 0
    if start_f1 is None:
        best_f1 = 0.0
    else:
        best_f1 = start_f1

    for i in range(start_epoch, epoch_num):
        gim.train()
        lrm.train()

        for j, batch in enumerate(iter):
            step += 1
            feature, label, masks = lrm(batch)
            masks = masks.view(-1)
            label = label.view(-1)[masks == 1].long()

            if args.aux_ce_reg and args.p2a_regularization:
                pred, x_emb, anchor_vecs, anchor_logits = gim(feature)
                pred = pred[masks == 1]
                loss = criterion(pred, label)
                aux_loss = aux_ce_loss(aux_criter, args.anchor_nums, anchor_logits)
                loss += args.lamda * aux_loss
                mse_losses = anchor_reg_loss(x_emb, label, anchor_vecs.view(-1, anchor_vecs.size(-1)))
                loss += args.p2a_coff * mse_losses
            elif args.aux_ce_reg and not args.p2a_regularization:
                pred, anchor_logits = gim(feature)
                pred = pred[masks == 1]
                loss = criterion(pred, label)
                aux_loss = aux_ce_loss(aux_criter, args.anchor_nums, anchor_logits)
                loss += args.lamda * aux_loss
            elif not args.aux_ce_reg and args.p2a_regularization:
                pred, x_emb, anchor_vecs = gim(feature)
                pred = pred[masks == 1]
                loss = criterion(pred, label)
                mse_losses = anchor_reg_loss(x_emb, label, anchor_vecs.view(-1, anchor_vecs.size(-1)))
                loss += args.p2a_coff * mse_losses
            else:
                pred = gim(feature)
                pred = pred[masks == 1]
                loss = criterion(pred, label)

            p, r, acc = accuracy(pred, label)

            optimizer.zero_grad() # 修改
            loss.backward()
            nn.utils.clip_grad_norm_(lrm.parameters(), 1.0)
            nn.utils.clip_grad_norm_(gim.parameters(), 1.0)

            optimizer.step()

            if scheduler:
                scheduler.step()

            if (j + 1) % log_freq == 0:
                logger.info(
                    'Train\tEpoch:[{:d}][{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(
                        i, j + 1, len(iter), loss, acc, p, r))

        if val_iter:
            f1_score = test_model(iter=val_iter, logger=logger, gim=gim, lrm=lrm, prefix='Val',
                      criterion=criterion, test_step=i + 1)
            if f1_score > best_f1:
                best_f1 = f1_score
                state = {
                    "lrm": lrm.state_dict(),
                    "gim": gim.state_dict(),
                    "epoch": i+1,
                    "val_f1": best_f1
                }
                torch.save(state, os.path.join(checkpoint_path, "{}_best.pth".format(fold)))
                logger.info("Val Best F1-score\t{:.4f}".format(best_f1))

    if test_iter:
        checkpoint = torch.load(os.path.join(checkpoint_path, "{}_best.pth".format(fold)))
        lrm.load_state_dict(checkpoint["lrm"])
        gim.load_state_dict(checkpoint["gim"])
        lrm = lrm.to(lrm.device)
        best_epoch = checkpoint["epoch"]
        val_f1 = checkpoint["val_f1"]
        logger.info("load from epoch {:d}  f1 score {:.4f}".format(best_epoch, val_f1))
        f1_score = test_model(iter=test_iter, logger=logger, gim=gim, lrm=lrm, prefix='Test',
                      criterion=criterion, test_step=i + 1)
        logger.info("Test F1 score\tEpoch\t{:d}\t{:.4f}".format(best_epoch, f1_score))
        f1_list.append(f1_score)


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
    parser.add_argument('--checkpoint_path', default="./saved_ckpt", type=str)
    parser.add_argument('--data_type', type=str, default='Wiki300')
    parser.add_argument('--model_name', default='GADE_300', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+')
    parser.add_argument('--gcn_layer', default=2, type=int)

    parser.add_argument('--anchor_nums', default=8, type=int)
    parser.add_argument('--num_pers', default=4, type=int)
    parser.add_argument('--p2a_coff', type=float, default=0.1)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--p2a_regularization', action='store_true', default=False)
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--aux_ce_reg', action='store_true', default=False)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lrm_type', default="albert", type=str)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gcn_dim = 768
    kfold = args.kfold

    params = args.__dict__
    with open(os.path.join(args.exp_dir, "training_params.txt"), 'w') as writer:
        writer.write(str(params))

    args.entity_path = 'datasets/' + args.data_type + '/target_entities.txt'
    args.data_path = 'datasets/' + args.data_type + '/TDD_dataset.json'
    args.description_path = 'datasets/' + args.data_type + '/entity_desc.json'
    ent_list = load_entity_list(args.entity_path)

    for i in range(kfold):

        # local relevance model
        if args.lrm_type == "roberta":
            lrm = LRM_Roberta(hid_dim=args.hid_dim, anchor_nums=args.anchor_nums, num_pers=args.num_pers, max_seq_length=args.max_seq_length, device=args.gpu)
        else:
            lrm = LRM_Albert(hid_dim=args.hid_dim, anchor_nums=args.anchor_nums, num_pers=args.num_pers, max_seq_length=args.max_seq_length, device=args.gpu)

        gim = AnchorGNN(gcn_dim, args.hid_dim, 2, args.gcn_layer, args.dropout, anchor_nums=args.anchor_nums, num_pers=args.num_pers, device=args.gpu, aux_ce=args.aux_ce_reg, p2a_reg=args.p2a_regularization)
        tokenizer = lrm.tokenizer
        input_tokens, label_inputs, desc_tokens = generate_data(ent_list, args.description_path,
                                                                args.data_path, tokenizer,
                                                                args.max_seq_length)
        # input_tokens, label_inputs, desc_tokens = generate_data_separate(ent_list, args.description_path,
        #                                                         args.data_path, tokenizer,
        #                                                         args.max_seq_length)     

        train_ent, val_ent, test_ent = get_kfold_data(ent_list, kfold, i)

        train_ent = yield_example(train_ent, input_tokens, label_inputs, desc_tokens)
        val_ent = yield_example(val_ent, input_tokens, label_inputs, desc_tokens)
        test_ent = yield_example(test_ent, input_tokens, label_inputs, desc_tokens)
        train_dataset = ComparisonDataset(train_ent)
        val_dataset = ComparisonDataset(val_ent)
        test_dataset = ComparisonDataset(test_ent)
        train_iter = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
        val_iter = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        test_iter = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in lrm.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.embed_lr},
            {'params': [p for n, p in lrm.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.embed_lr},
            {'params': [p for n, p in gim.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': [p for n, p in gim.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.lr}
        ]
    
        num_train_steps = len(train_iter) * args.epochs
        opt = AdamW(optimizer_grouped_parameters, eps=1e-8)
        scheduler = WarmupLinearSchedule(opt, warmup_steps=0, t_total=num_train_steps)

        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)
        # model_dir = args.exp_dir
        checkpoint_path = args.checkpoint_path + '/' + args.model_name
        log_dir = os.path.join(args.exp_dir, "logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        logger = set_logger(os.path.join(log_dir, str(time.time()) + "_" + args.model_name + ".log"))
        logger.info(params)
        logger.info("The {}-th fold training begins!".format(i))

        start_epoch = 0
        start_f1 = 0.0

        lrm = lrm.to(lrm.device)
        gim = gim.to(lrm.device)
        criterion = nn.CrossEntropyLoss().to(lrm.device)

        if args.aux_ce_reg:
            aux_criter = nn.CrossEntropyLoss().to(lrm.device)
        else:
            aux_criter = None
        
        train_model(train_iter, checkpoint_path, logger, i, gim, lrm, opt, criterion, aux_criter, args.epochs, test_iter=test_iter,
              val_iter=val_iter, scheduler=scheduler, log_freq=args.log_freq, start_epoch=start_epoch, start_f1=start_f1)


    logger.info("5 fold test f1-scores is {}".format(f1_list))
    logger.info("The average f1 score of 5 fold cross validation is {}".format(mean(f1_list)))
