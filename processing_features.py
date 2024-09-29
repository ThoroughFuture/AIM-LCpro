import pandas as pd
import numpy as np
import math
import os

import torch
import torch.nn.functional as f
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms

def OneHot(array, min, max):
    # The code min max corresponds to the upper and lower indexes
    pos_num = 0
    hot_num = [0, 0, 0, 0, 0, 0, 0, 2, 4, 3, 3, 4, 3, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 13, 10, 10, 9, 3, 3, 4, 7, 3,
               3, 18, 3]  # num class

    for k in range(min, max):

        arr = torch.from_numpy(array[:, k].astype(float)).to(torch.int64)
        if k == 7:
            arr -= 1  
            out = f.one_hot(arr)  # (sample_num, num_class=2)
            pos = np.zeros(shape=(1, out.shape[1]))  # (1, num_class)
        else:

            res = f.one_hot(arr, hot_num[k])
            out = np.concatenate((out, res), axis=1)  # along num_class cloumn cat
            pos = np.concatenate((pos, np.ones(shape=(1, res.shape[1])) * pos_num), axis=1)
        pos_num += 1
    return out, pos

def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)  # max_value != 1]

def get_train_data_normal():

    train_csv = pd.read_csv(r'/.csv')
    val_csv = pd.read_csv(r'/.csv')
    test_csv = pd.read_csv(r'/.csv')
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
        self.df = pd.read_csv(csv_path) 
        
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
                self.df = self.df[self.df['Should chemotherapy be given after surgery'] == 2]

            elif is_chemo==False:
                self.df = self.df[self.df['Should chemotherapy be given after surgery'] == 1]

        # data preprocess
        arr = Normalization_with_train(np.array(self.df), 37, 95, min_max_values)
        arr2, pos = OneHot(np.array(self.df), 7, 36)

        self.data = np.concatenate((arr2, arr), axis=1)  
        self.pos = np.concatenate((pos, np.arange(1, arr.shape[1] + 1) + np.max(pos).reshape(1, -1)), axis=1)  # 1*176

        for i in range(0, self.pos.shape[1]):  
            self.pos[0, i] = float(self.pos[0, i] - np.min(self.pos)) / (np.max(self.pos) - np.min(self.pos))

        self.label = self.df.iloc[:, :6]
        self.idx_list = self.df.index.to_list()  

        self.live_feature_path_dst = '/'
        self.die_feature_path_dst = '/'

    def __getitem__(self, item):
        # item 
        row_id = self.idx_list[item]  

        label = np.array(self.label)[item].astype(np.float64)

        five_live, five_live_time, five_die, five_die_time = label[2], label[3], label[4], label[5]

        # Classifier
        g_live = [normal_sampling(five_live_time, i) for i in range(-5, 65)]
        five_live_time_g_tmp = torch.tensor(g_live)
        five_live_time_g = f.normalize(five_live_time_g_tmp, p=1, dim=0)
        g_die = [normal_sampling(five_die_time, i) for i in range(-5, 65)]
        five_die_time_g_tmp = torch.tensor(g_die)
        five_die_time_g = f.normalize(five_die_time_g_tmp, p=1, dim=0)

        clinical_data = self.data[item].reshape(1, -1).astype(float)
        clinical_data = clinical_data + self.pos  # Add location information

        img_live_f = torch.load(os.path.join(self.live_feature_path_dst, self.mode+'_'+str(row_id)+'.pt'), map_location='cpu')[0]#[:, :256]
        img_die_f = torch.load(os.path.join(self.die_feature_path_dst, self.mode+'_'+str(row_id)+'.pt'), map_location='cpu')[0]#[:, :256]


        return (torch.squeeze(img_live_f), torch.squeeze(img_die_f)), clinical_data, five_live, five_live_time, five_live_time_g, five_die, five_die_time, five_die_time_g

    def __len__(self):
        return len(self.idx_list)

if __name__ == '__main__':
    train_dataset = data_read_cli_img_slice_feature(csv_path='train.csv',
                                           mode='train',
                                           live_time=False,
                                           die_time=False,
                                           chemo_flag=False,
                                           is_chemo=False,
                                           drop_pre=True)
    y = train_dataset[0]
