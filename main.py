#! /usr/bin/env python

import argparse
import json
from typing import List

import torch
import torch.utils.data as data
from blstm import BLSTM
from cache import load_cache, store_cache
from cgd import CGDDataset
from config import DefaultTrainConfig
from fit import Fitter

setting = json.load(open("config.json"))


def load_dataset(category: str) -> data.Dataset:
    cache_name = f"{category}_dataset"
    cgd_dataset = load_cache(cache_name)
    if cgd_dataset is None:
        cgd_dataset = CGDDataset(
            setting["dataset"][category]["cgd_filepath"],
            setting["hyperparameter"]["input_size"],
        )
        store_cache(cache_name, ["cgd.py", "config.json"], cgd_dataset)

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

    config = DefaultTrainConfig()

    parser = argparse.ArgumentParser(
        description="VulDeePecker implemented in pytorch. "
    )
    subparser = parser.add_subparsers(
        help="cmd", dest="cmd", description="Command to run. "
    )
    subparser.required = True

    train_group = subparser.add_parser("cv", help="Run cross validation. ")
    train_group.add_argument("category", choices=list(setting["dataset"].keys()))
    train_group.add_argument("-l", "--num-layers", nargs="?", type=int)
    train_group.add_argument("-d", "--dropout", nargs="?", type=float)
    train_group.add_argument("-b", "--batch-size", nargs="?", type=int)
    train_group.add_argument("-e", "--num-epochs", nargs="?", type=int)
    train_group.add_argument("-lr", "--learning-rate", nargs="?", type=float)
    train_group.add_argument("-k", "--k-fold", nargs="?", type=int)
    train_group.add_argument("-t", "--test-size", nargs="?", type=float)
    train_group.add_argument("-o", "--output", nargs="?", type=argparse.FileType("w"))

    result = parser.parse_args()
    cmd = result.cmd
    category = result.category

    if cmd == "cv":
        parameters = [
            "num_layers",
            "dropout",
            "batch_size",
            "num_epochs",
            "learning_rate",
            "k_fold",
            "test_size",
        ]
        for parameter in parameters:
            val = getattr(result, parameter)
            if val is not None:
                setattr(config, parameter, val)

        print("[*] loading dataset...")
        dataset = load_dataset(category)
        print("[+] loading dataset complete")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BLSTM(
            config.input_size,
            config.hidden_size,
            config.num_layers,
            config.num_classes,
            config.dropout,
            device,
        ).to(device)
        print("[*] training model...")
        fitter = Fitter(model, device, config)
        total_result = fitter.cross_validation(dataset)

        if result.output is None:
            f = open("./cross_val.csv", "w")
        else:
            f = result.output
        f.write(
            "fold,epoch,train_f1,val_f1,train_acc,val_acc,train_recall,val_recall,train_loss,val_loss\n"
        )
        for fold, fold_result in enumerate(total_result):
            for epoch, (
                train_summary_loss,
                train_total_score,
                val_summary_loss,
                val_total_score,
            ) in enumerate(fold_result):
                print(
                    fold + 1,
                    epoch + 1,
                    train_total_score.f1,
                    val_total_score.f1,
                    train_total_score.precision,
                    val_total_score.precision,
                    train_total_score.recall,
                    val_total_score.recall,
                    train_summary_loss.avg,
                    val_summary_loss.avg,
                    sep=",",
                    file=f,
                )
        f.close()

        print("[*] training model complete...")
