#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
sys.path.append('../../')

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
#from visdom import Visdom
#python -m visdom.server   启动服务器画图
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from itertools import zip_longest
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

#env = Visdom(use_incoming_socket=False)
#assert env.check_connection()
from Model.dataset.log import log_dataset
from Model.dataset.sample import sliding_window, session_window
from Model.tools.utils import (save_parameters, seed_everything,
                               train_val_split)

import psutil
from Model.tools.utils import *
from Model.tools.WordVector import *
from Model.models.mog_lstm_cnn2 import *
# from demo.mog_lstm_cnn_BGL import predict
from Model.tools.predict import Predicter

class Updater():
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']
        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']

        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']
        self.data_type=options['data_type']
        self.Event_TF_IDF=options['Event_TF_IDF']
        self.template=options['template']
        self.patience = options['patience']
        self.action=options['action']
        self.vbs_flag = options['vbs']
        self.fer_flag = options['fer']

        #超参数
        self.alpha = options['alpha']  #标签损失
        self.beta = options['beta']    #原始类别分数损失
        self.gama = options['gama']    #中间特征损失
        self.memory = options['mem']   #缓冲区大小
        self.vbs_lr = 0.005
        self.feature_lr=1

        self.model_path = options['model_path']
        self.update_model = options['update_model']
        self.options=options



        # self.input_size = options['input_size']
        # self.hidden_size = options['hidden_size']
        # self.num_layer = options['num_layers']
        # self.num_classes = options['num_classes']
        # self.cnn_length = options['cnn_length']
        # self.filter_num=options['filter_num']
        # self.filter_size=options['filter_size']



        os.makedirs(self.save_dir, exist_ok=True)#创建要保存的目录
        if self.sample == 'sliding_window':
          if self.data_type=="HDFS":
             train_logs, train_labels = sliding_window(self.data_dir,
                                                  datatype='train',
                                                  window_size=self.window_size,Event_TF_IDF=self.Event_TF_IDF,template=self.template)
             """val_logs, val_labels = sliding_window(self.data_dir,
                                              datatype='val',
                                              window_size=self.window_size,
                                              sample_ratio=0.001)"""
          elif self.data_type=="BGL":
              train_logs, train_labels = sliding_window(self.data_dir,
                                                        datatype='train',
                                                        window_size=self.window_size,data_type="BGL",Event_TF_IDF=self.Event_TF_IDF,template=self.template)
              """val_logs, val_labels = sliding_window(self.data_dir,
                                                    datatype='val',
                                                    window_size=self.window_size,
                                                    sample_ratio=0.001,data_type="BGL")"""


        elif self.sample == 'session_window':#根据会话窗口划分出训练测试日志集，包含 日志序列，计数特征，语义特征等key
            if self.data_type=="HDFS":
              train_logs, train_labels = session_window(self.data_dir,
                                                      datatype='train',Event_TF_IDF=self.Event_TF_IDF,template=self.template)
              val_logs, val_labels = session_window(self.data_dir,
                                                  datatype='val',Event_TF_IDF=self.Event_TF_IDF,template=self.template)
            elif self.data_type=="BGL":


                example_dir = self.data_dir+"example/"
                if self.fer_flag:
                    example_logs, example_labels, example_outputs, example_features= session_window(example_dir,
                                                         datatype='train',data_type="BGL",Event_TF_IDF=self.Event_TF_IDF,template=self.template, soft_flag=True,fer_flag=self.fer_flag)


                else:
                    example_logs, example_labels,example_outputs = session_window(example_dir,
                                                          datatype='train',data_type="BGL",Event_TF_IDF=self.Event_TF_IDF,template=self.template,soft_flag=True,fer_flag=self.fer_flag)

                update_dir = self.data_dir +"update/"
                update_logs, update_labels = session_window(update_dir,
                                                          datatype='train', data_type="BGL",
                                                          Event_TF_IDF=self.Event_TF_IDF, template=self.template)
                val_logs, val_labels = session_window(update_dir,
                                                      datatype='val',data_type="BGL",Event_TF_IDF=self.Event_TF_IDF,template=self.template)
            elif self.data_type=="OpenStack":
                train_logs, train_labels = session_window(self.data_dir,
                                                          datatype='train', data_type="OpenStack",
                                                          Event_TF_IDF=self.Event_TF_IDF, template=self.template)
            elif self.data_type=="Thunderbird":
                train_logs, train_labels = session_window(self.data_dir,
                                                          datatype='train', data_type="Thunderbird",
                                                          Event_TF_IDF=self.Event_TF_IDF, template=self.template)

        else:
            raise NotImplementedError
         #将日志数据规范化成log_dataset类 ，继承自dataset
        if self.fer_flag:
            example_dataset = log_dataset(logs=example_logs,
                                          labels=example_labels,
                                          outputs=example_outputs,
                                          features=example_features,
                                          seq=True,
                                          quan=self.quantitatives,
                                          sem=self.semantics,
                                          soft_flag=True,
                                          fer_flag=self.fer_flag)
        else:
            example_dataset = log_dataset(logs=example_logs,
                                          labels=example_labels,
                                          outputs=example_outputs,
                                          seq=True,
                                          quan=self.quantitatives,
                                          sem=self.semantics,
                                          soft_flag=True,
                                          fer_flag=self.fer_flag)

        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        update_dataset = log_dataset(logs=update_logs,
                                     labels=update_labels,
                                     seq=self.sequentials,
                                     quan=self.quantitatives,
                                     sem=self.semantics)

        del example_logs
        del val_logs

        del update_logs
        """del val_logs"""
        gc.collect() #垃圾回收
        #用dataloader加载数据集
        self.example_loader = DataLoader(example_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,#取出一批数据后洗牌
                                       pin_memory=True)

        self.update_loader = DataLoader(update_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=True,  # 取出一批数据后洗牌
                                        pin_memory=True)
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_example_log = len(example_dataset)

        self.num_update_log = len(update_dataset)

        self.num_valid_log = len(valid_dataset)

        """print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))"""
        print('Example batch size %d ,Update batch size %d' %
              (options['batch_size'], options['batch_size']))
     #选择加速硬件,model为设计好的深度学习模型
        self.model = model.to(self.device)
        #选择加速器
        if options['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),#传入模型参数
                                             lr=options['lr'],
                                             momentum=0.9)
        elif options['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=options['lr'],
                betas=(0.9, 0.999),
            )
        else:
            raise NotImplementedError

        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_score = -1
        save_parameters(options, self.save_dir + "parameters.txt") #将参数选项option保存下来
        self.log = { #记录训练过程
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "update": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "example": {key: []
                       for key in ["epoch", "lr", "time", "loss"]}
        }
        if options['resume_path'] is not None:
            if os.path.isfile(options['resume_path']):
                self.resume(options['resume_path'], load_optimizer=True)
            else:
                print("Checkpoint not found")

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.log = checkpoint['log']
        #    self.best_f1_score = checkpoint['best_f1_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "log": self.log,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_vbs" + str(suffix) + ".pth"
        torch.save(checkpoint, save_path)
        print("Save model checkpoint at {}".format(save_path))

    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")


    def train(self, epoch):

        self.log['example']['epoch'].append(epoch)#增加日志信息
        self.log['update']['epoch'].append(epoch)  # 增加日志信息
        start = time.strftime("%H:%M:%S")#打印时间
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: update | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['example']['lr'].append(lr)
        self.log['example']['time'].append(start)
        self.log['update']['lr'].append(lr)
        self.log['update']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()#梯度置0
        criterion = nn.CrossEntropyLoss()#选择损失函数
        tbar1 = tqdm(self.example_loader,desc="bar1",leave=False)#进度条展示数据加载  读取数据
        tbar2 = tqdm(self.update_loader,desc="bar2",leave=False)  # 进度条展示数据加载  读取数据

        exa_num_batch = len(self.example_loader)#一批的个数
        upd_num_batch = len(self.update_loader)
        example_total_losses = 0
        update_total_losses=0

        old_iter_count = 0
        new_iter_count = 0
        #
        # tbar = tqdm(self.example_loader, desc="\r")
        # for i, (log, label) in enumerate(tbar):
        #     features = []
        #     for value in log.values():
        #         features.append(value.clone().to(self.device))
        #     output = self.model(features=features, device=self.device)
        # Model = MogLSTM(in_size2=300, input_sz=self.input_size, hidden_sz=self.hidden_size, mog_iteration=2,cnn_length=self.cnn_length,filter_num=self.filter_num,filter_sizes=self.filter_size)
        # Model.load_state_dict(torch.load(self.model_path)['state_dict'])
        # model=Model.to(self.device)
        # model.eval()

        for i, (t1, t2) in enumerate(zip_longest(tbar1, tbar2)):
            # 解包 t1 和 t2
            if self.fer_flag:
                log1, label1, soft_label1, convs = t1 if t1 else (None, None, None,None)
            else:
                log1, label1, soft_label1 = t1 if t1 else (None, None, None)
            log2, label2 = t2 if t2 else (None, None)

            if log1 is not None:
                old_iter_count += 1
                features1 = []
                features_onelist = []
                sequentials = log1['Sequentials'].squeeze() # 获取 'Sequentials' 的值
                # torch.set_printoptions(threshold=float("inf"), edgeitems=10, linewidth=150)
                # print(sequentials)
                # seq_list.append(sequentials.clone())
                for value in log1['Semantics']:
                    features1.append(value.clone().to(self.device))
                features = torch.stack(features1)  # 32个（50，768）张量的list
                features_onelist.append(features)
                if self.fer_flag:
                    if self.vbs_flag:
                        output1, vbs_loss, inter = self.model(features=features_onelist, device=self.device)
                    else:
                        output1, inter = self.model(features=features_onelist, device=self.device)
                else:
                    output1 = self.model(features=features_onelist, device=self.device)
                # test= model(features=features_onelist, device=self.device)
                # soft_label = self.softlabel[(old_iter_count - 1) * self.batch_size:old_iter_count * self.batch_size]
                soft_label = torch.tensor([[float(num) for num in label.split(',')] for label in soft_label1])


                if self.fer_flag:
                    loss = self.alpha * criterion(output1, label1.to(self.device)) \
                           + self.beta * F.mse_loss(output1, soft_label.to(self.device))
                    if self.vbs_flag:
                        loss += self.vbs_lr * vbs_loss
                    ferloss=F.mse_loss(inter[0], convs['conv1'].to(self.device))\
                            +F.mse_loss(inter[1], convs['conv2'].to(self.device))\
                            +F.mse_loss(inter[2], convs['conv3'].to(self.device))
                    loss += ferloss * self.gama
                else:
                    if self.update_model == 'derpp':
                        loss = self.alpha * criterion(output1, label1.to(self.device)) \
                               + self.beta * F.mse_loss(output1, soft_label.to(self.device))
                    else:
                        loss = F.mse_loss(output1, soft_label.to(self.device))

                example_total_losses += float(loss)
                loss /= self.accumulation_step
                loss.backward()
                # 梯度优化！！！
                if (i + 1) % self.accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                tbar1.set_description("Train loss: %.5f" % (example_total_losses / (old_iter_count)))
                # tbar1.update(1)
                # 使用 tqdm.write 输出描述信息，避免覆盖
                # tqdm.write("tbar1 Train loss: %.5f" % (example_total_losses / old_iter_count))


                # print()  # 添加换行符
            if log2 is not None:
                new_iter_count+=1
                features2 = []
                for value in log2.values():  # 读取特征
                    features2.append(value.clone().detach().to(self.device))
                    # 特征输入模型，训练得output
                """print(" features[0].shape",features[0].shape,'\n')
                print(features[0])
                print(" features[1].shape", features[0].shape, '\n')
                print(features[1])"""
                # print(" features[0]", features[0], '\n')
                # print("label:",label.numpy().tolist())
                if self.vbs_flag:
                    output, vbs_loss,inter = self.model(features=features2, device=self.device)
                else:
                    output,inter = self.model(features=features2, device=self.device)
                # 计算损失函数
                # print('output.shape',output.shape,'\n')
                # print('label.shape',label.shape,'\n')
                # print("output:", output)
                # print("label:", label)
                loss = criterion(output, label2.to(self.device))
                # print('label.shape', label.shape, '\n')
                if self.vbs_flag:
                    loss += self.vbs_lr * vbs_loss

                update_total_losses += float(loss)
                loss /= self.accumulation_step
                loss.backward()
                # 梯度优化！！！
                if (i + 1) % self.accumulation_step == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                tbar2.set_description("Train loss: %.5f" % (update_total_losses / (new_iter_count)))
                # tbar2.update(1)  # 更新进度条
                # tqdm.write("tbar2 Train loss: %.5f" % (update_total_losses / new_iter_count))



                # x=epoch
        # y=total_losses / num_batch
        """
        env2.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=pane1,  # win参数确认使用哪一个pane
            update='append')  # 我们做的动作是追加
            """
        self.log['example']['loss'].append(example_total_losses / exa_num_batch)
        self.log['update']['loss'].append(update_total_losses / upd_num_batch)

    def valid(self, epoch):
        self.model.eval()
        self.log['valid']['epoch'].append(epoch)
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        self.log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        self.log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                if self.vbs_flag:
                    output, vbs_loss, inter = self.model(features=features, device=self.device)
                else:
                    output,inter= self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        # if total_losses / num_batch < self.best_loss:
        #     self.best_loss = total_losses / num_batch
        #     self.save_checkpoint(epoch,
        #                          save_optimizer=False,
        #                          suffix="bestloss")

        return total_losses / num_batch

    def example_agre(self):
        # file_path="../data/BGL/MLog_log_train.csv"

        # 读取数据
        input_file_path_1 = '../data/update/BGL/MLog_log_train.csv'
        input_file_path_2 = '../data/example/BGL/MLog_log_train.csv'

        # Load the CSV files into DataFrames
        df1 = pd.read_csv(input_file_path_1)
        df2 = pd.read_csv(input_file_path_2)

        # Merge the two DataFrames
        data= pd.concat([df1, df2], ignore_index=True)

        # 提取用于聚类的特征列（假设为 'Sequence' 列）
        # 将字符串序列转换为数值特征向量
        sequences = data['Sequence'].apply(lambda x: np.fromstring(x, sep=' '))

        # 找到最长的序列长度
        max_length = sequences.apply(len).max()

        # 将所有序列填充到相同长度
        sequences_padded = np.array([np.pad(seq, (0, max_length - len(seq)), 'constant') for seq in sequences])

        # 标准化数据
        scaler = StandardScaler()
        sequences_scaled = scaler.fit_transform(sequences_padded)

        # 使用轮廓系数确定聚类数量
        silhouette_scores = []
        k_range = range(2, 11)  # 尝试 2 到 10 个聚类
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(sequences_scaled)
            silhouette_avg = silhouette_score(sequences_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            # 找到具有最高轮廓系数的聚类数量
        best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
        print(f"最佳聚类数量为: {best_k}")

        # 进行聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        data['Cluster'] = kmeans.fit_predict(sequences_scaled)

        # 计算需要提取的样本数量
        total_samples = len(data)
        example_sample_count = self.memory

        # 按照每个聚类中样本个数的比例提取范例样本
        example_samples = pd.DataFrame()
        for cluster in range(best_k):
            cluster_data = data[data['Cluster'] == cluster]
            cluster_sample_count = int(len(cluster_data) / total_samples * example_sample_count)
            example_samples = example_samples.append(cluster_data.sample(cluster_sample_count, random_state=42))

            # 创建输出目录
        output_dir = '../data/example/BGL'
        os.makedirs(output_dir, exist_ok=True)

        # 只保留指定的四列
        example_samples = example_samples[['Index', 'Sequence', 'label', 'Start_time', 'End_time']]

        # 保存范例样本
        output_file_path = os.path.join(output_dir, 'MLog_log_train.csv')
        example_samples.to_csv(output_file_path, index=False)

        print(f"范例样本已保存到 {output_file_path}")

        # old_preds = old_preds.detach().clone()

    def soft_label(self):

        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.model.eval()
        # self.model.set_vbs(False)
        predict_list = []
        label_list = []
        print('model_path: {}'.format(self.model_path))

        example_dir = self.data_dir + "example/"
        example_logs, example_labels = session_window(example_dir,
                                                      datatype='train', data_type="BGL", Event_TF_IDF=self.Event_TF_IDF,
                                                      template=self.template)

        example_dataset = log_dataset(logs=example_logs,
                                      labels=example_labels,
                                      seq=True,
                                      quan=self.quantitatives,
                                      sem=self.semantics)

        example_output_loader = DataLoader(example_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,  # 取出一批数据后洗牌
                                           pin_memory=True)

        tensor_list = []
        inter_list=[]
        seq_list = []
        tbar = tqdm(example_output_loader, desc="\r")
        for i, (log, label) in enumerate(tbar):
            features = []
            features_onelist=[]
            sequentials = log['Sequentials'].squeeze()  # 获取 'Sequentials' 的值
            seq_list.append(sequentials.clone())
            for value in log['Semantics']:
                features.append(value.clone().to(self.device))
            features = torch.stack(features)  # 32个（50，768）张量的list
            features_onelist.append(features)


            output, inter = self.model(features=features_onelist, device=self.device,layer_feature=True)


            tensor_list.append(output.detach().clone())
            inter_list.append(inter)

        # 存储三个卷积层的输出
        conv1, conv2, conv3 = [], [], []
        for inter in inter_list:
            conv1.append(inter[0])
            conv2.append(inter[1])
            conv3.append(inter[2])
        # sequences = torch.cat(seq_list, dim=0)
        old_preds = torch.cat(tensor_list, dim=0)
        inter_conv1 = torch.cat(conv1, dim=0)
        inter_conv2 = torch.cat(conv2, dim=0)
        inter_conv3 = torch.cat(conv3, dim=0)


        output_column = [f"{row[0]},{row[1]}" for row in old_preds]

        # 将 old_inter 转换为字符串列表，每个张量用逗号分隔
        inter_column1 = [','.join(map(str, row.tolist())) for row in inter_conv1]
        inter_column2 = [','.join(map(str, row.tolist())) for row in inter_conv2]
        inter_column3 = [','.join(map(str, row.tolist())) for row in inter_conv3]

        # 指定 CSV 文件路径
        file_path = '../data/example/BGL/MLog_log_train.csv'

        # 创建 DataFrame
        output_df = pd.DataFrame({
            'output': output_column,
            'conv1': inter_column1,
            'conv2': inter_column2,
            'conv3': inter_column3

        })

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取现有数据
            existing_df = pd.read_csv(file_path)
            # 将新列追加到现有 DataFrame
            existing_df['output'] = output_column
            existing_df['conv1'] = inter_column1
            existing_df['conv2'] = inter_column2
            existing_df['conv3'] = inter_column3
            # 将更新后的 DataFrame 写入 CSV 文件
            existing_df.to_csv(file_path, index=False)
        else:
            # 如果文件不存在，直接写入新 DataFrame
            output_df.to_csv(file_path, index=False)

        print(f'Data appended to {file_path} successfully.')


    def log_memory_usage(self,file_path, message):
         with open(file_path, 'a') as f:
             f.write(message + '\n')

    def start_update(self):
        x, y = 0, 0
        """env2 = Visdom(use_incoming_socket=False)
        pane1 = env2.line(
            X=np.array([x]),
            Y=np.array([y]),
            opts=dict(title='loss function'))
        """
        self.model_path = "../result/mog_lstm_cnn/mog_lstm_cnn_vbs" + str(self.vbs_flag) + ".pth"
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)


        #train()
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio

            if epoch in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,35,37,38,39,40,42,43,44,45,46,47,48,50]:
                if self.options['vbs']:
                    self.options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsTrue.pth"
                else:
                    self.options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsFalse.pth"
                predicter = Predicter(self.model, self.options)
                predicter.predict_supervised()

            log_file = '../result/mog_lstm_cnn/memory_usage_log.txt'
            self.log_memory_usage(log_file, f"Epoch {epoch}")

            memory_info_start = psutil.virtual_memory()
            start_usage = memory_info_start.percent
            self.log_memory_usage(log_file, f"开始前 - 内存使用率: {start_usage}%")

            self.train(epoch)

            # 查看每个epoch结束后的内存使用率
            memory_info_end = psutil.virtual_memory()
            end_usage = memory_info_end.percent
            self.log_memory_usage(log_file, f"结束后 - 内存使用率: {end_usage}%")

            # 计算内存使用率差值
            usage_difference = end_usage - start_usage
            self.log_memory_usage(log_file, f"内存使用率差值: {usage_difference}%")
            val_loss = self.valid(epoch)
            save_path = self.save_dir + self.model_name + "_vbs" + str(self.vbs_flag) + ".pth"



            early_stopping(epoch, val_loss, self.model, save_path, self.action, self.optimizer, self.log)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            """if epoch >= self.max_epoch // 2 and epoch % 2 == 0:
                self.valid(epoch)
                self.save_checkpoint(epoch,
                                     save_optimizer=True,
                                     suffix="epoch" + str(epoch))"""
            # self.save_checkpoint(epoch, save_optimizer=True, suffix=self.vbs_flag)
            self.save_log()




        self.example_agre()
        self.soft_label()


