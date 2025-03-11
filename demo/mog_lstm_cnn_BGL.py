#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')


from Model.tools.predict import Predicter
from Model.tools.train import Trainer
from Model.tools.update import Updater
from Model.tools.utils import *
from Model.tools.WordVector import *
from Model.models.mog_lstm_cnn2 import *
# Config Parameters

options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cuda"

# Smaple
options['sample'] = "session_window"
options['window_size'] = -1

# Features
options['sequentials'] = False
options['quantitatives'] = False
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 768
options['hidden_size'] = 512
options['num_layers'] = 1
options['num_classes'] = 2
options['cnn_length']=2
# Train

options['batch_size'] = 32
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.0005  #train:0.001
options['max_epoch'] = 60
options['lr_step'] = (40, 50)
options['lr_decay_ratio'] = 0.1

options['resume_path'] =   None# "../result/mog_lstm_cnn/mog_lstm_cnn_last.pth"

options['model_name'] = "mog_lstm_cnn"
options['save_dir'] = "../result/mog_lstm_cnn/"
options['Event_TF_IDF']=1
options['template']="../data/BGL/BGL.log_templates.csv"
# Predict
options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_last.pth"
options['num_candidates'] = -1
options['data_type']='BGL'
options['filter_num']=16
options['filter_size']='3,4,5'
options['patience']=5
options['action']="update"  #train or update
options['update_model']="fer" #der or derpp or fer
options['vbs']=False  #true or false
options['fer']=True  #true or false
#超参数
options['mem']=800
options['alpha']=0.4
options['beta']=0.4
options['gama']=0.2

seed_everything(seed=1234)
#input_size = 512
#hidden_size = 512
vocab_size = 2
batch_size = 4
lr = 3e-3
mogrify_steps = 5        # 5 steps give optimal performance according to the paper
dropout = 0.5            # for simplicity: input dropout and output_dropout are 0.5. See appendix B in the paper for exact values
tie_weights = True       # in the paper, embedding weights and output weights are tied
betas = (0, 0.999)       # in the paper the momentum term in Adam is ignored
weight_decay = 2.5e-4    # weight decay is around this value, see appendix B in the paper
clip_norm = 10           # paper uses cip_norm of 10
batch_sz, seq_len, feat_sz, hidden_sz = 5, 10, 32, 16

def train():
    Model =MogLSTM(in_size2=300,input_sz=options['input_size'], hidden_sz=options['hidden_size'],mog_iteration=2,cnn_length=options["cnn_length"],filter_num=options['filter_num'],filter_sizes=options['filter_size'],vbs_flag=options['vbs'],fer_flag=options['fer'],device=options['device'])
    trainer = Trainer(Model, options)
    trainer.start_train()


def update():
    Model =MogLSTM(in_size2=300,input_sz=options['input_size'], hidden_sz=options['hidden_size'],mog_iteration=2,cnn_length=options["cnn_length"],filter_num=options['filter_num'],filter_sizes=options['filter_size'],vbs_flag=options['vbs'],fer_flag=options['fer'],device=options['device'])
    if options['vbs']:
        options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsTrue.pth"
    else:
        options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsFalse.pth"

    exa_dire=options['data_dir']+'update/BGL/MLog_valid_exa.csv'
    upd_dire = options['data_dir'] + 'update/BGL/MLog_log_valid.csv'
    output_dire = options['data_dir'] + 'update/BGL/MLog_log_valid.csv'

    # 读取CSV文件
    exa_df = pd.read_csv(exa_dire, usecols=['Index', 'Sequence', 'label', 'Start_time', 'End_time'])
    upd_df = pd.read_csv(upd_dire, usecols=['Index', 'Sequence', 'label', 'Start_time', 'End_time'])

    # 合并两个DataFrame
    merged_df = pd.concat([exa_df, upd_df], ignore_index=True)

    # 将合并后的DataFrame写入新的CSV文件
    merged_df.to_csv(output_dire, index=False)
    merged_df.to_csv(exa_dire, index=False)

    print(f'Merged data has been saved to {output_dire}')
    print(f'Merged data has been saved to {exa_dire}')


    updater = Updater(Model, options)
    updater.start_update()


def predict():
    Model =  MogLSTM(in_size2=300,input_sz=options['input_size'], hidden_sz=options['hidden_size'],mog_iteration=2,cnn_length=options["cnn_length"],filter_num=options['filter_num'],filter_sizes=options['filter_size'],vbs_flag=options['vbs'],device=options['device'])
    if options['vbs']:
        options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsTrue.pth"
    else:
        options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsFalse.pth"
    predicter = Predicter(Model, options)
    predicter.predict_supervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
   # parser.add_argument('mode', choices=['train', 'predict'])
    args = parser.parse_args()
   # if args.mode == 'train':

    if options['action']== 'update':
        predict()
        update()
        predict()
    else:
        # predict()
        train()

        predict()
