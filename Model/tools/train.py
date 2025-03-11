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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import psutil
#env = Visdom(use_incoming_socket=False)
#assert env.check_connection()
from Model.dataset.log import log_dataset
from Model.dataset.sample import sliding_window, session_window
from Model.tools.utils import *
from Model.tools.predict import Predicter
class Trainer():
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

        self.patience=options['patience']

        self.action=options['action']
        self.vbs_flag=options['vbs']
        self.fer_flag = options['fer']
        self.vbs_lr = 0.0001
        self.model_path = "../result/mog_lstm_cnn/mog_lstm_cnn_vbs" + str(self.fer_flag) + ".pth"

        self.options = options

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

                train_logs, train_labels = session_window(self.data_dir,
                                                          datatype='train',data_type="BGL",Event_TF_IDF=self.Event_TF_IDF,template=self.template)
                val_logs, val_labels = session_window(self.data_dir,
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
        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)



        del train_logs

        del val_logs
        gc.collect() #垃圾回收
        #用dataloader加载数据集
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,#取出一批数据后洗牌
                                       pin_memory=True)

        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       pin_memory=True)

        self.num_train_log = len(train_dataset)
        if self.action == "update":
            self.update_train_log = len(train_dataset)

        """self.num_valid_log = len(valid_dataset)"""

        """print('Find %d train logs, %d validation logs' %
              (self.num_train_log, self.num_valid_log))"""
        print('Train batch size %d ,Validation batch size %d' %
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

        # self.model_path=save_path


    def save_log(self):
        try:
            for key, values in self.log.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Log saved")
        except:
            print("Failed to save logs")

    def train(self, epoch):
        self.log['train']['epoch'].append(epoch)#增加日志信息
        start = time.strftime("%H:%M:%S")#打印时间
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
              (epoch, start, lr))
        self.log['train']['lr'].append(lr)
        self.log['train']['time'].append(start)
        self.model.train()
        self.optimizer.zero_grad()#梯度置0
        criterion = nn.CrossEntropyLoss()#选择损失函数
        tbar = tqdm(self.train_loader, desc="\r")#进度条展示数据加载  读取数据
        num_batch = len(self.train_loader)#一批的个数
        total_losses = 0
        for i, (log, label) in enumerate(tbar):
            #print ("log:\n",log,"\n");
            #print ("log.value:\n",log.values(),"\n");
            features = []
            for value in log.values():#读取特征
               features.append(value.clone().detach().to(self.device))
                #特征输入模型，训练得output
            """print(" features[0].shape",features[0].shape,'\n')
            print(features[0])
            print(" features[1].shape", features[0].shape, '\n')
            print(features[1])"""
            #print(" features[0]", features[0], '\n')
            #print("label:",label.numpy().tolist())
            if self.vbs_flag:
                output,vbs_loss,inter = self.model(features=features, device=self.device)
            else:
                output= self.model(features=features, device=self.device)

            #计算损失函数
            # print('output.shape',output.shape,'\n')
            #print('label.shape',label.shape,'\n')
            #print("output:", output)
            #print("label:", label)

            loss = criterion(output, label.to(self.device))
            #print('label.shape', label.shape, '\n')
            if self.vbs_flag:
                loss += self.vbs_lr*vbs_loss
            total_losses += float(loss)
            loss /= self.accumulation_step
            loss.backward()
            #梯度优化！！！
            if (i + 1) % self.accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))
        x=epoch
        y=total_losses / num_batch
        """
        env2.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=pane1,  # win参数确认使用哪一个pane
            update='append')  # 我们做的动作是追加
            """
        self.log['train']['loss'].append(total_losses / num_batch)

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
                    output = self.model(features=features, device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        self.log['valid']['loss'].append(total_losses / num_batch)

        return total_losses/num_batch

        # if total_losses / num_batch < self.best_loss:
        #     self.best_loss = total_losses / num_batch
        #     self.save_checkpoint(epoch,
        #                          save_optimizer=False,
        #                          suffix="bestloss")

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
        inter_list = []
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

            output,inter= self.model(features=features_onelist, device=self.device,layer_feature=True)
            tensor_list.append(output.detach().clone())
            inter_list.append(inter)

        #存储三个卷积层的输出
        conv1, conv2, conv3 = [], [], []
        for inter in inter_list:
            conv1.append(inter[0])
            conv2.append(inter[1])
            conv3.append(inter[2])


        sequences = torch.cat(seq_list, dim=0)
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

        # 创建 DataFrame，只包含二列
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

    def example_agre(self):
        # file_path="../data/BGL/MLog_log_train.csv"

        # 读取数据
        input_file_path = '../data/BGL/MLog_log_train.csv'
        data = pd.read_csv(input_file_path)

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
        example_sample_count = int(total_samples * 0.3)

        # 按照每个聚类中样本个数的比例提取范例样本
        example_samples = pd.DataFrame()
        for cluster in range(best_k):
            cluster_data = data[data['Cluster'] == cluster]
            cluster_sample_count = int(len(cluster_data) / total_samples * example_sample_count)
            example_samples = example_samples.append(cluster_data.sample(cluster_sample_count, random_state=42))

            # 创建输出目录
        output_dir = '../data/example/BGL'
        os.makedirs(output_dir, exist_ok=True)

        # 保存范例样本
        output_file_path = os.path.join(output_dir, 'MLog_log_train.csv')
        example_samples.to_csv(output_file_path, index=False)

        print(f"范例样本已保存到 {output_file_path}")
        # old_preds = old_preds.detach().clone()



    def log_memory_usage(self,file_path, message):
         with open(file_path, 'a') as f:
             f.write(message + '\n')


    def start_train(self):
        x, y = 0, 0
        """env2 = Visdom(use_incoming_socket=False)
        pane1 = env2.line(
            X=np.array([x]),
            Y=np.array([y]),
            opts=dict(title='loss function'))
        """
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio

            if epoch != 0:
                if self.options['vbs']:
                    self.options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsTrue.pth"
                else:
                    self.options['model_path'] = "../result/mog_lstm_cnn/mog_lstm_cnn_vbsFalse.pth"

                predicter = Predicter(self.model, self.options)
                predicter.predict_supervised()


            log_file = '../result/mog_lstm_cnn/memory_usage_log.txt'
            self.log_memory_usage(log_file, f"Epoch {epoch}")

            # 查看训练开始前的内存使用率
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

            val_loss=self.valid(epoch)
            save_path = self.save_dir + self.model_name + "_vbs" + str(self.vbs_flag) + ".pth"

            early_stopping(epoch,val_loss, self.model, save_path, self.action,self.optimizer,self.log)
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



