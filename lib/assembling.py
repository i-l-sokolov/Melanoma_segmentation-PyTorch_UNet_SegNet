import os
from glob import glob
import pickle
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from IPython.display import clear_output
import pandas as pd
from dataset import SegDataset
from losses import iou_pytorch, bce_loss, dice_loss, focal_loss, get_tversky
from training_eval import train, eval_epoch
from models import UNet, SegNet
import argparse

parser = argparse.ArgumentParser(description='Parser for training SegNet or UNet segmentation network')
parser.add_argument('--model', type=int, default=0, help='the number of model from the table')
parser.add_argument('--loss', type=int, default=0, help='the loss function from the table')
parser.add_argument('--batch_size', type=int, default=32, help='the size of batch during training')
parser.add_argument('--epochs', type=int, default=20, help='the number of epoch to train')
parser.add_argument('--new_dataframe', type=int, default=0, help='if non zero it will create new dataframe with results')
parser.add_argument('--save_model', type=int, default=0, help='if non zero - the best model during training will be saved')
parser.add_argument('--evaluate', type=int, default=0, help='If non zero - it will evaluate model on test dataset at the end of training. Works with save_model equal to 1')

args = parser.parse_args()

model_n = args.model
loss = args.loss
batch_size = args.batch_size
epochs = args.epochs
new_df = args.new_dataframe
save_model = args.save_model
evaluate = args.evaluate


losses = [nn.BCELoss(), bce_loss, dice_loss, focal_loss]
losses += [get_tversky(i / 10) for i in range(11)]
losses_names = ['BCELoss', 'bce_loss', 'dice_loss', 'focal_loss']
losses_names += [f'tversky_loss_alpha_{i / 10:.2f}' for i in range(11)]

models = [SegNet(), UNet(mode='pool'), UNet(mode='stride'), UNet(mode='dilate')]
models_names = ['SegNet', 'UNet', 'UNet_stride', 'UNet_dilate']

criterion = losses[loss]
criterion_name = losses_names[loss]

model_name = models_names[model_n]
model = models[model_n]

os.chdir('../')

torch.manual_seed(42)
np.random.seed(42)

with open('pickles/split', 'rb') as f:
    split = pickle.load(f)
train_folders = split['train']
valid_folders = split['valid']
test_folders = split['test']

train_dataset = SegDataset(train_folders, 'train')
valid_dataset = SegDataset(valid_folders,'valid')
test_dataset = SegDataset(test_folders,'test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=25, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=25, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')


if __name__ == '__main__':
    df = train(model, model_name,
           train_dataloader, valid_dataloader, valid_dataset,
           epochs, optimizer,
           criterion, criterion_name,
           scheduler,
           iou_pytorch, device, save_model)
    if new_df:
        df.to_csv('results/results.csv', index=False)
    else:
        try:
            df_exist = pd.read_csv('results/results.csv')
            df = pd.concat([df_exist, df])
            df.to_csv('results/results.csv', index=False)
        except:
            df.to_csv('results/results.csv', index=False)
    if evaluate:
        if os.path.exists(f'results/{model_name}_{criterion_name}.pth'):
            model.load_state_dict(torch.load(f'results/{model_name}_{criterion_name}.pth'))
            _, iou_test = eval_epoch(model,criterion,iou_pytorch,test_dataloader,device)
            print(f'The IoU for test dataset is {iou_test:.4f}')
        else:
            print('There is no saved model. Most likely the argument save_model set to 0')
