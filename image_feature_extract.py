import os
import torch
from torchvision import transforms
from PIL import Image
import resnet
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel as DP
import torch.multiprocessing as mp
from dataset import dataload as DataLoad
import resnet
import utils
import utils.roc as roc
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('-seed', '--seed', type=int, default=3047)
parser.add_argument('-amp', '--amp', type=bool, default=True)
parser.add_argument('-input_size', type=int, default=2048)
parser.add_argument('-patch_size', type=int, default=256)

args = parser.parse_args()

patch_row = int(args.input_size /args. patch_size)  
patch_num = patch_row * patch_row

gpu = 1
flag = 'die'

torch.cuda.set_device(gpu) 

model = resnet.ResNet50().cuda()
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DP(model, device_ids=[gpu], output_device=[gpu])
if flag == 'live':
    mdoel_pt_path = 'pt'  # Five-year progress
else:
    mdoel_pt_path = 'pt'  # Five-year deaths
model.load_state_dict(torch.load(mdoel_pt_path, map_location='cpu'))

model.eval()

# Defining Data Conversions
t = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.22)], p=0.4),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
            # transforms.Resize((2048, 2048)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

# Enter the directory where the image is located
input_directory = ''  # externel val
# Directory where output features are saved
if flag == 'live':
    output_directory = "five_live"
else:
    output_directory = "five_die"
os.makedirs(output_directory, exist_ok=True)

# Iterate through the image files in the directory
pbar = tqdm(total=len(os.listdir(input_directory)), unit='img')
batch_n = 1

for filename in os.listdir(input_directory):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  
        # Build the save path
        save_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.pt")

        if not(os.path.exists(save_path)):
        
            image_path = os.path.join(input_directory, filename)
            input_tensor = t(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda()
            image_input = slice_image_batch(input_tensor, patch_num, args)
            with torch.no_grad():
                res_logits = model(image_input)

            
            sm = torch.softmax(res_logits[0], dim=-1)  # (batch_size * patch, 2)
            result_id = torch.tensor([], dtype=torch.int64).cuda()
            for z in range(batch_n):
                patch_256_i = sm[z*patch_num:(z+1)*patch_num, :]
                first_idx = z*patch_num
                result_id_patch = get_id(patch_256_i) + first_idx
                result_id = torch.cat((result_id, result_id_patch), dim=0)
            sep = int(result_id.shape[0] / batch_n)

            feature = torch.zeros(size=(result_id.shape[0], 2048)).cuda()
            feature_ave = torch.zeros(size=(batch_n, 2048)).cuda()
            result_p = torch.zeros(size=(result_id.shape[0], 1)).cuda()
            result_p_avg = torch.zeros(size=(batch_n, 1)).cuda()
            ln_feature = res_logits[1] # shape(256: b*64, 268, 1, 1)

            for j in range(result_id.shape[0]):
                feature[j] = ln_feature[result_id[j], ...].squeeze()
                result_p[j] = sm[result_id[j], 1]

            for b in range(batch_n):
                feature_ave[b] = torch.mean(feature[b * sep : (b + 1) * sep], dim=0)

            result_p_avg = torch.mean(result_p)

            target = (feature_ave.cpu(), result_p_avg.cpu())

            # Preservation of Characteristics
            torch.save(target, save_path)

            print(f"Saved features for {filename} to {save_path}")

        pbar.update(1)
