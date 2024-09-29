# vit mlp both
import matplotlib.pyplot as plt
import datetime

import numpy as np
import math
import tqdm
from tqdm import tqdm
import math
import sys
import warnings
import pandas as pd 
import random

import os
import sys
import warnings
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

# from module import Head_regressor, Head_classifier, ComplexHeadRegressor, Head_regressor_with_dropout
import utils.roc as roc
from utils import *

class Head_classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(3072, 70)
        self.ReLu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tmp_futrue):
        # tmp_futrue = self.dropout(tmp_futrue)
        out = self.linear1(tmp_futrue)
        out_s = self.softmax(out)
        out_sn = F.normalize(out_s, p=1, dim=1)
        return out_sn
    
class Head_regressor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.8)
        self.linear1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tmp_futrue):
        # tmp_futrue = self.dropout(tmp_futrue)
        out = self.sigmoid(self.bn1(self.linear1(tmp_futrue)))
        out_sn = self.sigmoid(self.linear2(out))
        return out_sn
    

class Head_regressor_0523(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(p=0.8)
        self.linear1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tmp_futrue):
        tmp_futrue = self.dropout(tmp_futrue)
        out = self.sigmoid(self.bn1(self.linear1(tmp_futrue)))
        out_sn = self.sigmoid(self.linear2(out))
        return out_sn

class mlp_five_live_die_cla_1024_cat_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(200, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1536, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.linear3 = nn.Linear(1536, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.live_linear1 = nn.Linear(3072, 512)
        self.live_bn1 = nn.BatchNorm1d(512)
        self.live_final = nn.Linear(512, 2)
        self.die_linear1 = nn.Linear(3072, 512)
        self.die_bn1 = nn.BatchNorm1d(512)
        self.die_final = nn.Linear(512, 2)
        
        self.dropout = nn.Dropout(p=0.5)
        self.ReLu = nn.ReLU()

    def forward(self, clinical_data, img_live, img_die):
        out = self.ReLu(self.bn1(self.linear1(clinical_data)))
        img_live = self.ReLu(self.bn2(self.linear2(img_live)))
        img_die = self.ReLu(self.bn3(self.linear3(img_die)))
        
        fused_futrue = torch.cat((out, img_live, img_die), dim=1)
        # fused_futrue = self.dropout(fused_futrue)
        five_live = self.ReLu(self.live_bn1(self.live_linear1(fused_futrue)))
        five_die = self.ReLu(self.die_bn1(self.die_linear1(fused_futrue)))

        five_live = self.live_final(five_live)
        five_die = self.die_final(five_die)
  
        return five_live, five_die, fused_futrue

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum() / loss.shape[0]
    return loss

    
def OneHot(array, min, max):
    # The code min max corresponds to the upper and lower indexes
    pos_num = 0
    hot_num = [0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 3, 4, 3, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 13, 10, 10, 9, 3, 3, 4, 7, 3,
               3, 18, 3]  # num class

    for k in range(min, max):

        arr = torch.from_numpy(array[:, k].astype(float)).to(torch.int64)
        if k == 7:
            arr -= 1  
            out = F.one_hot(arr)  # (sample_num, num_class=2)
            pos = np.zeros(shape=(1, out.shape[1]))  # (1, num_class)
        else:

            res = F.one_hot(arr, hot_num[k])
            out = np.concatenate((out, res), axis=1)  # along num_class cloumn cat
            pos = np.concatenate((pos, np.ones(shape=(1, res.shape[1])) * pos_num), axis=1)
        pos_num += 1
    return out, pos

def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)  # max_value != 1]

def get_train_data_normal():

    train_csv = pd.read_csv(r'/train.csv')
    val_csv = pd.read_csv(r'/val.csv')
    test_csv = pd.read_csv(r'/test.csv')
    combined_df = pd.concat([train_csv, val_csv, test_csv], ignore_index=True)
    combined_data = np.array(combined_df)

    # Calculate the minimum and maximum values for each column
    min_max_values = {}
    zero_columns = []  # Used to store the serial numbers of all 0 columns
    for col_index in range(37, 95):
        column = combined_data[:, col_index].astype(float)
        min_value = np.min(column)
        max_value = np.max(column)
        min_max_values[col_index] = {'min': min_value, 'max': max_value}
        
        # If both the minimum and maximum values are 0, the serial number of the column is recorded
        if min_value == 0 and max_value == 0:
            zero_columns.append(col_index)
    return min_max_values, zero_columns


def Normalization_with_train(array, min, max, min_max_values):
    # The normalized min max corresponds to the upper and lower indexes
    elp = 1e-6
    for k in range(min, max):
        arr = array[:, k].astype(float).copy()
        for i in range(arr.shape[0]):
            # arr[i] = float(arr[i] - np.min(arr)) / (np.max(arr) - np.min(arr))
            minima = min_max_values[k]['min']
            maxima = min_max_values[k]['max']
            arr[i] = float(arr[i] - min_max_values[k]['min'] + elp) / (min_max_values[k]['max'] - min_max_values[k]['min'] + elp)
        array[:, k] = arr
    return array[:, min:max]

class data_read_cli_img_slice_feature(Dataset):
    def __init__(self, csv_path, mode, live_time=False, die_time=False, chemo_flag=False, is_chemo=False, drop_pre=False):
        
        self.mode = mode
        # 7-35onehot 
        self.df = pd.read_csv(csv_path) # encoding='gbk'
        
        if drop_pre:
            self.df['Post relapse treatment'] = 0

        # prepare for normalization
        min_max_values, _ = get_train_data_normal()

        if live_time or die_time:
            if die_time:
                self.df = self.df[self.df['camel2Within 5 years of time of death'] < 60]
            if live_time:
                self.df = self.df[self.df['camel2Time to progress within 5 years'] < 60]
            # self.df = self.df.reset_index(drop=True)
                
        if chemo_flag:
            #  Does not require postoperative chemotherapy: 1, Requires postoperative chemotherapy: 2
            if is_chemo==True:
                self.df = self.df[self.df['Should chemotherapy be given after surgery'] == 2]   #(23, 95)

            elif is_chemo==False:
                self.df = self.df[self.df['Should chemotherapy be given after surgery'] == 1]
        print(self.df.shape)
        # data preprocess
        arr = Normalization_with_train(np.array(self.df), 37, 95, min_max_values)
        arr2, pos = OneHot(np.array(self.df), 7, 36)

        self.data = np.concatenate((arr2, arr), axis=1)  
        self.pos = np.concatenate((pos, np.arange(1, arr.shape[1] + 1) + np.max(pos).reshape(1, -1)), axis=1)  

        for i in range(0, self.pos.shape[1]):  
            self.pos[0, i] = float(self.pos[0, i] - np.min(self.pos)) / (np.max(self.pos) - np.min(self.pos))

        self.label = self.df.iloc[:, :6]
        self.id = self.df.iloc[:, 6]
        self.idx_list = self.df.index.to_list()  

        self.live_feature_path_dst = 'five_live/'  # internel
        self.die_feature_path_dst = 'five_die/'



    def __getitem__(self, item):
        # item 
        row_id = self.idx_list[item]  
        label = np.array(self.label)[item].astype(np.float64)
        # id = np.array(self.id)[item].astype(np.float64)
        id = np.array(self.id)[item]
        three_live, three_live_time, five_live, five_live_time, five_die, five_die_time = label[0], label[1], label[2], label[3], label[4], label[5]

        # Classifier
        g_live = [normal_sampling(five_live_time, i) for i in range(-5, 65)]
        five_live_time_g_tmp = torch.tensor(g_live)
        five_live_time_g = F.normalize(five_live_time_g_tmp, p=1, dim=0)
        g_die = [normal_sampling(five_die_time, i) for i in range(-5, 65)]
        five_die_time_g_tmp = torch.tensor(g_die)
        five_die_time_g = F.normalize(five_die_time_g_tmp, p=1, dim=0)

        clinical_data = self.data[item].reshape(1, -1).astype(float)
        clinical_data = clinical_data + self.pos  # Add location information
 
        img_live_f = torch.load(os.path.join(self.live_feature_path_dst, self.mode+'_'+str(row_id)+'.pt'), map_location='cpu')[0]
        img_die_f = torch.load(os.path.join(self.die_feature_path_dst, self.mode+'_'+str(row_id)+'.pt'), map_location='cpu')[0]

        return (torch.squeeze(img_live_f), torch.squeeze(img_die_f)), clinical_data, five_live, five_live_time, five_live_time_g, five_die, five_die_time, five_die_time_g, id, three_live, three_live_time

    def __len__(self):
        return len(self.idx_list)

warnings.simplefilter("ignore")

def predict():
    global print_df
    rank = torch.Tensor([i for i in range(70)]).cuda()

    mlp_model.eval()
    head_live.eval()
    head_die.eval()

    dataloader = test_data
    dataset_num = n_test          

    sample_num = 0  # sample_num

    three_live_correct = 0
    five_live_correct = 0
    five_die_correct = 0
    three_live_TP = 0
    five_live_TP = 0
    five_die_TP = 0
    three_live_FP = 0
    five_live_FP = 0
    five_die_FP = 0
    three_live_l1 = 0
    five_live_l1 = 0
    five_die_l1 = 0

    pbar = tqdm(total=n_test, unit='img')

    for i, data in enumerate(dataloader, 0):

        img_live_feature, img_die_feature = data[0]
        img_live_feature = img_live_feature.cuda()
        img_die_feature = img_die_feature.cuda()

        clinical_data = data[1].cuda().view(data[1].shape[0], -1).to(torch.float32)
        label_five_live = data[2].cuda()
        label_five_live_time = data[3].cuda().view(-1, 1)
        label_five_live_time_g = data[4].cuda()
        label_five_die = data[5].cuda()
        label_five_die_time = data[6].cuda().view(-1, 1)
        label_five_die_time_g = data[7].cuda()
        patient_id = data[8].cuda()
        label_three_live = data[9].cuda()
        label_three_live_time = data[10].cuda().view(-1, 1)

        batch_n = label_five_live.shape[0]    #1

        with torch.no_grad():

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):

                out = mlp_model(clinical_data, img_live_feature, img_die_feature) 
                # out = mlp_model(img_live_feature) 
                # print(f"tmp logits shape: {tmp_logits.shape}")
                out_five_live_time_g = head_live(out[2]) 
                out_five_die_time_g = head_die(out[2])

            if mode == 'cla':
                out_five_live_time = torch.sum(out_five_live_time_g * rank, dim=1)
                out_five_die_time = torch.sum(out_five_die_time_g * rank, dim=1)
            else:
                out_five_live_time = torch.floor(out_five_live_time_g.to(torch.float64) * 60 + 0.5)
                out_five_die_time = torch.floor(out_five_die_time_g.to(torch.float64) * 60 + 0.5)

            out = list(out)
            out[0] = torch.softmax(out[0], dim=1)
            out[1] = torch.softmax(out[1], dim=1)
            out = tuple(out)
            
            for batch in range(batch_n):  
                if out[0][batch, 1].item() >= args.val_cutoff_live:
                    five_live = 1
                    five_live_time = out_five_live_time.item()
                else:
                    five_live = 0
                    five_live_time = 60#out_five_live_time.item()

                if out[1][batch, 1].item() >= args.val_cutoff_die:
                    five_die = 1
                    five_die_time = out_five_die_time.item()
                else:
                    five_die = 0
                    five_die_time = 60#out_five_die_time.item()

        pbar.update(1)

        print_df.loc[i, 'camel2Lung cancer progression within 5 years_pred'] = out[0][batch, 1].item()
        print_df.loc[i, 'camel2Death within 5 years due to lung cancer_pred'] = out[1][batch, 1].item()  

        # df.loc[i, 'pathology number'] = patient_id.cpu().numpy()
        df.loc[i, 'pathology number'] = patient_id                       
        df.loc[i, 'camel2Lung cancer progression within 5 years'] = five_live
        df.loc[i, 'camel2Time to progress within 5 years'] = five_live_time
        df.loc[i, 'camel2Death within 5 years due to lung cancer'] = five_die
        df.loc[i, 'camel2Within 5 years of time of death'] = five_die_time

        if five_live_time < 36:
            df.loc[i, 'camel2Lung cancer progression within 3 years'] = 1
            df.loc[i, 'camel2Time to progress within 3 years'] = five_live_time

        else:
            df.loc[i, 'camel2Lung cancer progression within 3 years'] = 0
            df.loc[i, 'camel2Time to progress within 3 years'] = 36

        if label_three_live.item() == df.loc[i, 'camel2Lung cancer progression within 3 years']:
            three_live_correct += 1
            if df.loc[i, 'camel2Lung cancer progression within 3 years'] == 1:
                three_live_TP += 1
        elif df.loc[i, 'camel2Lung cancer progression within 3 years'] == 1:
            three_live_FP += 1
        
        if label_five_live.item() == five_live:
            five_live_correct += 1
            if five_live == 1:
                five_live_TP += 1
        elif five_live == 1:
                five_live_FP += 1

        if label_five_die.item() == five_die:
            five_die_correct += 1
            if five_die == 1:
                five_die_TP += 1
        elif five_die == 1:
                five_die_FP += 1

        three_live_l1 += torch.sum(torch.abs(df.loc[i, 'camel2Time to progress within 3 years'] - label_three_live_time))
        five_live_l1 += torch.sum(torch.abs(five_live_time - label_five_live_time))
        five_die_l1 += torch.sum(torch.abs(five_die_time - label_five_die_time))

        # statistics
        sample_num += batch_n  

    print('——————————————————————————————————————————————————————————————————————————————————————————————————————')

    three_live_l1_avg = three_live_l1 / sample_num
    five_live_l1_avg = five_live_l1 / sample_num
    five_die_l1_avg = five_die_l1 / sample_num

    print(f"sample num: {sample_num}")

    print(f'three live precision: {three_live_correct/sample_num}, five live precision: {five_live_correct/sample_num}, five die precision: {five_die_correct/sample_num} \n' 
          f'three live TPR: {three_live_TP/three_live_p}, five live TPR: {five_live_TP/five_live_p}, five die TPR: {five_die_TP/five_die_p} \n' 
          f'three live FPR: {three_live_FP/three_live_n}, five live FPR: {five_live_FP/five_live_n}, five die FPR: {five_die_FP/five_die_n} \n' 
          f'three live time l1(avg): {three_live_l1_avg}, five live time l1(avg): {five_live_l1_avg}, five die time l1(avg): {five_die_l1_avg}')
    
    df.to_csv(os.path.join(args.result_path, '912inference_cutoff_internel_result.csv'), encoding='gbk')
    
    #The results predicted by the model, roc Fig.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='batch')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='threading')

    parser.add_argument('-seed', '--seed', type=int, default=3047, help='random seed')
    parser.add_argument('-amp', '--amp', type=bool, default=True, help='Mixing accuracy')

    parser.add_argument('-gpu', '--gpu', default=[7], help='GPU')

    parser.add_argument('--val_cutoff_live', type=float, default=0)  
    parser.add_argument('--val_cutoff_die', type=float, default=0)  
    
    mode = 'reg'
    # class_ = 'die'
    parser.add_argument('--program_suffix', type=str, default='0912/test_all')
    parser.add_argument('--result_path', type=str, default='/result')

    args = parser.parse_args()

    args.result_path = os.path.join(args.result_path, args.program_suffix)
    os.makedirs(args.result_path, exist_ok=True)
    
    
    torch.cuda.set_device('cuda:{}'.format(args.gpu[0]))

    # dataset
   
    test_dataset = data_read_cli_img_slice_feature(csv_path='/home/zengjunyang/to_zjy/chest_lym/internel_val/internel_val.csv', 
                                           mode='internel_test',   ###externel_test  #internel_test  
                                           live_time=False,
                                           die_time=False,
                                           chemo_flag=False, 
                                           is_chemo=True,
                                           drop_pre=True)
   
    n_test = len(test_dataset)    
    test_data = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False, 
        pin_memory=True,
        drop_last=False)  
    
    print(f"test data size: {n_test}")   #23

    # model

    mlp_model = mlp_five_live_die_cla_1024_cat_3().cuda()
    mlp_model_pt_path = '/traindata/xiongjiahang/my_chest/checkpoint/0513/1024_fold_val_all_81_lr6/mlp_121_37.pt'
    mlp_model.load_state_dict(torch.load(mlp_model_pt_path, map_location='cpu'))

    if mode == 'cla':
        head_live = Head_classifier().cuda()
        head_die = Head_classifier().cuda()
    else:
        head_live = Head_regressor().cuda()
        head_die = Head_regressor().cuda()
        # head = Head_regressor().cuda()
        # head = ComplexHeadRegressor().cuda()
        # head = Head_regressor_with_dropout().cuda()

    live_head_cla_path = ''
    head_live.load_state_dict(torch.load(live_head_cla_path, map_location='cpu'))
    die_head_cla_path = ''
    head_die.load_state_dict(torch.load(die_head_cla_path, map_location='cpu'))

    ce_loss = nn.CrossEntropyLoss().cuda()
    l1_loss = nn.L1Loss().cuda()
    mse_loss = nn.MSELoss().cuda()  
    
    label_df = pd.read_csv('csv')
    
    print_df = label_df[['camel2Lung cancer progression within 5 years', 'camel2Death within 5 years due to lung cancer']]
    print_df['camel2Lung cancer progression within 5 years_pred'] = None
    print_df['camel2Death within 5 years due to lung cancer_pred'] = None 

    print((label_df['camel2Lung cancer progression within 3 years'] == 1).sum())
    print((label_df['camel2Lung cancer progression within 5 years'] == 1).sum())
    print((label_df['camel2Died within 5 years of lung cancer亡'] == 1).sum())

    three_live_p = (label_df['camel2Lung cancer progression within 3 years'] == 1).sum()
    five_live_p = (label_df['camel2Lung cancer progression within 5 years'] == 1).sum()
    five_die_p = (label_df['camel2Death within 5 years due to lung cancer'] == 1).sum()

    three_live_n = (label_df['camel2Lung cancer progression within 3 years'] == 0).sum()
    five_live_n = (label_df['camel2Lung cancer progression within 5 years'] == 0).sum()
    five_die_n = (label_df['camel2Death within 5 years due to lung cancer'] == 0).sum()
    predict()
