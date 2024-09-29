import sys
import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.utils.data.distributed
import sys
import warnings
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from tqdm import tqdm
import cv2 as cv
from camel.model.resnet import ResNet18
from camel.train.camel_frame import train
import numpy as np
import matplotlib.pyplot as plt
from camel.dataload.DataLoad_inference import data_load_feature_mlp
from camel.utils import slice_image
from camel.utils import roc
from Moe import MoE
from model import all_512_pfs,Dino_Mlp
from sklearn.metrics import roc_auc_score
warnings.simplefilter("ignore")
sys.setrecursionlimit(10000000)



def roc(y_true, y_score,sklearn=True):

    pos_label =1
    # Count the number of positive and negative samples
    num_positive_examples = (y_true == pos_label).sum() 
    num_negtive_examples = len(y_true) - num_positive_examples
    tp, fp = 0, 0
    tpr, fpr, thresholds = [], [], []
    score = max(y_score) + 1

    # Calculate fpr and tpr based on the sorted prediction scores, respectively
    for i in np.flip(np.argsort(y_score)):
        # Handling the case where the samples have the same predicted scores
        if y_score[i] != score:
            fpr.append(fp / num_negtive_examples)
            tpr.append(tp / num_positive_examples)
            thresholds.append(score)
            score = y_score[i]

        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / num_negtive_examples)
    tpr.append(tp / num_positive_examples)
    thresholds.append(score)

    maxindex = (np.array(tpr) - np.array(fpr)).tolist().index(max(np.array(tpr) - np.array(fpr)))
    cutoff = thresholds[maxindex]  
    index = thresholds.index(cutoff)

    se = tpr[index]  
    sp = 1-fpr[index]  

    if sklearn:
        auc = roc_auc_score(y_true,y_score)
    else:
        auc = 0
        for i in range(len(tpr) - 1):
            auc += (fpr[i + 1] - fpr[i]) * tpr[i + 1]

    correct = 0
    for i in range(y_score.shape[0]):
        if y_score[i] >= cutoff and y_true[i] == 1:
            correct += 1
        elif y_score[i] < cutoff and y_true[i] == 0:
            correct += 1
    acc = correct / y_score.shape[0]*100

    return auc,se,sp,index,fpr,tpr,cutoff,acc

def auc_figue(auc,se,sp,index,fpr,tpr,cutoff,acc,type):

    fig, ax = plt.subplots()
    plt.plot([0, 1], '--')
    plt.plot(fpr[index], tpr[index], 'bo')
    ax.text(fpr[index], tpr[index] + 0.02, f'cut_off={round(cutoff,3)}', fontdict={'fontsize': 10})
    plt.plot(fpr, tpr)
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    text = f'AUC:{round(auc, 3)}\nSE:{round(se,3)}\nSP:{round(sp,3)}\nAccuracy:{round(acc, 3)}%\n'
    ax.text(0.6, 0.05, text, fontsize=12)


def main(model,train_path,test_path,BatchSize,lr=1e-3,half=False,label=None,num_work=8):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    pt_path = os.listdir(train_path)
    num = len(pt_path)

    ture = []
    camel_result = []

    print('Load data')
    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
            
        camel_sm = torch.load(f'{train_path}{pt_path[i]}',map_location=torch.device('cpu')).view(-1,2108,1,1)
        
        for kk in range(0,camel_sm.shape[0]):
            camel_result.append(camel_sm[kk])
            label = int(pt_path[i][0])
            
            ture.append(label)
    
    dataset = data_load_feature_mlp(camel_result,ture)
    inference_data = DataLoader(
                                dataset=dataset,
                                batch_size=BatchSize,
                                shuffle=True,  
                                num_workers=num_work,
                                pin_memory=True,
                                drop_last=False)  

    pt_path = os.listdir(test_path)
    num = len(pt_path)

    ture = []
    camel_result = []

    print('Load data')
    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        
        camel_sm = torch.load(f'{test_path}{pt_path[i]}',map_location=torch.device('cpu')).view(-1,2108,1,1)
        
        for kk in range(0,camel_sm.shape[0]):
            camel_result.append(camel_sm[kk])
            label = int(pt_path[i][0])
            
            ture.append(label)

    dataset = data_load_feature_mlp(camel_result,ture)
    inference_data_test = DataLoader(
                                dataset=dataset,
                                batch_size=BatchSize,
                                shuffle=True,  
                                num_workers=num_work,
                                pin_memory=True,
                                drop_last=False)  
                            
    ce_loss = nn.CrossEntropyLoss().cuda()


    print('Start training')

    if half:
        model = model.half()
    best_auc = 0
    for epoch in range(0,100): 
        model.train()
        for i, data in enumerate(tqdm(inference_data, 0, leave=False, ncols=70)):

            if half:
                In = data[0].half().cuda()  # half
                label = data[1].half().cuda()
            else:
                In = data[0].cuda()  
                label = data[1].cuda()
            
            optimizer.zero_grad()
            out= model(In.squeeze(0))
        
            sm = torch.softmax(out, dim=-1)

            loss = ce_loss(out,label)

            loss.backward()
                
            optimizer.step()
            #print(loss.item())

            pred_label = []
            pred_1 = []
            for lb in range(label.shape[0]):
                pred_label.append(label[lb].item())
                pred_1.append(sm[lb, 1].item())
            pred_label = torch.tensor(pred_label).cuda()
            pred_1 = torch.tensor(pred_1).cuda()

            if i == 0:
                pred_roc = pred_1.detach().cpu().numpy()
                label_roc = pred_label.detach().cpu().numpy()
            else:
                pred_roc = np.concatenate((pred_roc, pred_1.detach().cpu().numpy().reshape(-1)), axis=0)
                label_roc = np.concatenate((label_roc, pred_label.detach().cpu().numpy().reshape(-1)), axis=0)

        model.eval()
        for i, data in enumerate(tqdm(inference_data_test, 0, leave=False, ncols=70)):
            # if i % 2 != 0: continue
            if half:
                In = data[0].half().cuda()  # half
                label = data[1].half().cuda()
            else:
                In = data[0].cuda()  
                label = data[1].cuda()
            In[:,:2049,0,0] = 0
            out= model(In.squeeze(0))
        
            sm = torch.softmax(out, dim=-1)

            pred_label = []
            pred_1 = []
            for lb in range(label.shape[0]):
                pred_label.append(label[lb].item())
                pred_1.append(sm[lb, 1].item())
            pred_label = torch.tensor(pred_label).cuda()
            pred_1 = torch.tensor(pred_1).cuda()

            if i == 0:
                pred_roc_test = pred_1.detach().cpu().numpy()
                label_roc_test = pred_label.detach().cpu().numpy()
            else:
                pred_roc_test = np.concatenate((pred_roc_test, pred_1.detach().cpu().numpy().reshape(-1)), axis=0)
                label_roc_test = np.concatenate((label_roc_test, pred_label.detach().cpu().numpy().reshape(-1)), axis=0)
        auc_train,se_train,sp_train,index_train,fpr_train,tpr_train,cutoff_train,acc_train = roc(label_roc, pred_roc)
        auc_test,se_test,sp_test,index_test,fpr_test,tpr_test,cutoff_test,acc_test = roc(label_roc_test, pred_roc_test)

        print(f'train:{auc_train},test:{auc_test}')
        
        if auc_test>best_auc :
            best_auc  = auc_test
            print(f'>>>>>>>>>>>>>>>>>>>test:{auc_test}<<<<<<<<<<<<<<<<<<<<<<<<')

            # auc_figue(auc_test,se,sp,index,fpr,tpr,cutoff,acc)
            auc_figue(auc_train,se_train,sp_train,index_train,fpr_train,tpr_train,cutoff_train,acc_train,'train')
            auc_figue(auc_test,se_test,sp_test,index_test,fpr_test,tpr_test,cutoff_test,acc_test,'test')
            torch.save(model.state_dict(), f"{epoch}.pt")

if __name__ == '__main__':

    torch.cuda.set_device(0)
    
    dist.init_process_group(backend='nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.random.manual_seed(3047)

    model = MoE(input_channel=2108, input_size=[1, 1], hidden_size=64, output_channel=2, num_experts=8,noisy_gating=True, k=1).cuda()
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)


    train_path = f''

    test_path = f''
    main(model,train_path=train_path,test_path=test_path,BatchSize=256,lr=1e-5,half=False,label=None,num_work=8)
       
