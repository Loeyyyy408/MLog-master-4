import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))


# https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


# https://blog.csdn.net/folk_/article/details/80208557
def train_val_split(logs_meta, labels, val_ratio=0.1):
    total_num = len(labels)
    train_index = list(range(total_num))
    train_logs = {}
    val_logs = {}
    for key in logs_meta.keys():
        train_logs[key] = []
        val_logs[key] = []
    train_labels = []
    val_labels = []
    val_num = int(total_num * val_ratio)

    for i in range(val_num):
        random_index = int(np.random.uniform(0, len(train_index)))
        for key in logs_meta.keys():
            val_logs[key].append(logs_meta[key][random_index])
        val_labels.append(labels[random_index])
        del train_index[random_index]

    for i in range(total_num - val_num):
        for key in logs_meta.keys():
            train_logs[key].append(logs_meta[key][train_index[i]])
        train_labels.append(labels[train_index[i]])

    return train_logs, train_labels, val_logs, val_labels

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=-0.01):  #train: 0.03  vbstrain,vbsupdate的后俩个日期，update:0.01    vbsupdate:-0.005
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self,epoch, val_loss, model, path,action,optimizer,log):
        score = -val_loss
        # #update终止条件
        if action == 'update':
            if epoch == 10:  # update
                self.delta = -0.05
            if epoch == 20:
                self.delta = 0


        #train终止条件
        if action == 'train':
            if epoch == 10:  # update: -0.005  vbsupdate的后俩个日期   update取10 -0.005
                self.delta = -0.02  # 10-16：-0.01
            #     # self.patience = 8
            # if epoch == 30:  # update:0   update取20
            #     self.delta = -0.01  # 10-16：-0.005
            if epoch == 30:  # update:0   update取20
                self.delta = 0  # 10-16：-0.005




        # if epoch == 20:
        #     self.delta =-0.02   #  train:-0.02
        # if epoch == 40:
        #     self.delta = 0  # train:0


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch,val_loss, model, path,optimizer,log)
        elif score >= self.best_score + self.delta:

            self.best_score = score
            self.save_checkpoint(epoch, val_loss, model, path, optimizer, log)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    # def save_checkpoint(self, epoch,val_loss, model, path):
    #     if self.verbose:
    #         print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
    #         torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
    #     self.val_loss_min = val_loss

    def save_checkpoint(self, epoch, val_loss,model,path,optimizer,log,save_optimizer=True):
        if self.verbose:
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_loss": self.val_loss_min,
                "log": log,
                "best_score": self.best_score
            }
            if save_optimizer:
                checkpoint['optimizer'] = optimizer.state_dict()
            save_path = path
            torch.save(checkpoint, save_path)
            print("Save model checkpoint at {}".format(save_path))
        self.val_loss_min = val_loss
