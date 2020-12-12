#! /usr/bin/env python

from typing import List

import torch
from cgd import CGDDataset
import os
from cache import load_cache, store_cache
import torch.utils.data as data
from blstm import BLSTM
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score
import torch.nn as nn
import argparse
import json
from sklearn.model_selection import KFold
import numpy as np
from fit import Fitter
from config import DefaultTrainConfig

setting = json.load(open('config.json'))

def load_dataset(category: str) -> data.Dataset:
    cache_name = f'{category}_dataset'
    cgd_dataset = load_cache(cache_name)
    if cgd_dataset is None:
        cgd_dataset = CGDDataset(setting['dataset'][category]['cgd_filepath'], setting['hyperparameter']['input_size'])
        store_cache(cache_name, ['cgd.py', 'config.json'], cgd_dataset)
    
    return cgd_dataset

def calc_weight(dataset):
    total_step = len(dataset)
    weight = [0, 0]
    for _, label in dataset:
        weight[label] += 1

    weight[0], weight[1] = weight[1], weight[0]
    return torch.tensor(list(map(lambda v: v / total_step, weight)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VulDeePecker implemented in pytorch. ')
    parser.add_argument('cmd', choices=['train'])
    parser.add_argument('category', choices=list(setting['dataset'].keys()))
    result = parser.parse_args()

    cmd = result.cmd
    category = result.category

    config = DefaultTrainConfig()
    
    if cmd == 'train':
        print('[*] loading dataset...')
        dataset = load_dataset(category)
        print('[+] loading dataset complete')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BLSTM(config.input_size, config.hidden_size, config.num_layers, config.num_classes, config.dropout, device).to(device)
        print('[*] training model...')
        fitter = Fitter(model, device, config)
        total_result = fitter.cross_validation(dataset)

        with open('./cross_val.csv', 'w') as f:
            f.write('fold,epoch,train_f1,val_f1,train_acc,val_acc,train_recall,val_recall,train_loss,val_loss\n')
            for fold, fold_result in enumerate(total_result):
                for epoch, (train_summary_loss, train_total_score, val_summary_loss, val_total_score) in enumerate(fold_result):
                    print(fold+1, epoch+1, train_total_score.f1, val_total_score.f1, train_total_score.precision, val_total_score.precision, train_total_score.recall, val_total_score.recall, 
                        train_summary_loss.avg, val_summary_loss.avg, sep=',', file=f)

        print('[*] training model complete...')
        print(result)
    
