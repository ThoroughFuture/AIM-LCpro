import sys
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import roc_auc_score
from camel.eval import range_threshold


def roc(y_true, y_score,sklearn=True):
    pos_label =1
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

def auc_figue(auc,se,sp,index,fpr,tpr,cutoff,acc):

    fig, ax = plt.subplots()
    plt.plot([0, 1], '--')
    plt.plot(fpr[index], tpr[index], 'bo')
    ax.text(fpr[index], tpr[index] + 0.02, f'cut_off={round(cutoff,8)}', fontdict={'fontsize': 10})
    plt.plot(fpr, tpr)
    plt.axis("square")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC")
    text = f'AUC:{round(auc, 3)}\nSE:{round(se,3)}\nSP:{round(sp,3)}\nAccuracy:{round(acc, 3)}%\n'
    ax.text(0.6, 0.05, text, fontsize=12)
   
    plt.savefig(f'AUC.png')


data_path = f''
pt_path = os.listdir(data_path)
num = len(pt_path)

auc_best = 0
cutoff = 0.998
top= 3
threshold = range_threshold(start=0,end=cutoff,step=0.01)
ture = []
camel_result = []
patient_probabilities = []  
print('Load data')
for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        
    camel_sm = torch.load(f'{data_path}{pt_path[i]}',map_location=torch.device('cpu'))
    camel_result.append(camel_sm)
    label = int(pt_path[i][0])
    ture.append(label)
ture = np.array(ture)

print('seek to solve (an equation)')
for j in range(0,len(threshold)):
    print(f'{j}/{len(threshold)}')
    pred = []
    for i in tqdm(range(0, num), 0, leave=False, ncols=70):
        camel_sm = camel_result[i]     
        neg_sm = []
        pos_sm = []

        for k in range(0, camel_sm.shape[0]):

            if camel_sm[k].item() < (cutoff-threshold[j]):
                neg_sm.append(camel_sm[k].item())

            elif camel_sm[k].item() >= (cutoff+threshold[j]):
                pos_sm.append(camel_sm[k].item())
        
        if len(pos_sm) > len(neg_sm):
            if pos_sm:
                pred_sm = np.sort(pos_sm,axis=0)[-top:].mean()
            else:
                pred_sm = 0    #np.nan
            pred.append(pred_sm)
        else:
            if neg_sm:
                pred_sm = np.sort(neg_sm, axis=0)[:top].mean()
            else:
                pred_sm = 0
            pred.append(pred_sm)
    
    pred = np.array(pred)
    if np.isnan(pred).any():
        continue
    auc,se,sp,index,fpr,tpr,cutoff,acc = roc(ture, pred)  
    print(f'auc:{round(auc,5)},threshold:{threshold[j]}')
    huanzhe = (pt_path[i].split('_')[1]).split('.pt')[0]
    patient_probabilities.append((huanzhe, pred))
    if auc > auc_best:
        print(f'>>>>>>bestauc:{round(auc,5)},threshold:{threshold[j]}<<<<<<')
        #print(auc,se,sp,index,fpr,tpr,cutoff)
        auc_figue(auc,se,sp,index,fpr,tpr,cutoff,acc)
        auc_best = auc
        np.save('true.npy',ture)
        np.save('pred.npy',pred)


