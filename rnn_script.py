import os
import numpy as np
import sys
import torch
import torchvision 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from matplotlib import pyplot as plt
import math
from torch.autograd import grad
from collections import namedtuple
import argparse
import json

# import pascanu regularizer
modify_loss_path = '/Users/ebrahimfeghhi/NCEL/rnn_modeling/modify_loss/'
sys.path.insert(1, modify_loss_path)
from pascanu import *

parser = argparse.ArgumentParser(description='Select Dataset')
parser.add_argument('--data_code', type=int, default=0, help='Dataset code')
parser.add_argument('--dale', type=int, default=0, help='1 for Dale')
parser.add_argument('--pascanu', type=int, default=0, help='1 for pascanu')
parser.add_argument('--num_trials', type=int, default=int(1000), help='number of trials')
parser.add_argument('--n_in', type=int, default=int(50), help='number of input neurons')

args = parser.parse_args() 

data_code = args.data_code
dale = args.dale
pascanu = args.pascanu
num_trials=args.num_trials
n_in = args.n_in

# import dataset class 
dataset_folder_path = '/Users/ebrahimfeghhi/NCEL/rnn_modeling/datasets/'
sys.path.insert(1, dataset_folder_path)
from retrieve_datasets import retrieve
training_set = retrieve(data_code, num_trials, n_in)

# import loss class
if data_code==0: 
    dataset_folder_path = '/Users/ebrahimfeghhi/NCEL/rnn_modeling/modify_loss/'
    sys.path.insert(1, dataset_folder_path)
    from retrieve_loss import retrieve
    loss_params = {'resp_dur':training_set.resp_dur} # pass any params needed for custom loss 
    criterion = retrieve(data_code, loss_params) 
else:
    criterion=nn.L1Loss()

# import model class 
rnn_folder_path = '/Users/ebrahimfeghhi/NCEL/rnn_modeling/models/'
sys.path.insert(1, rnn_folder_path)
from Neuro_RNN_JIT import NeuroRNNCell, NeuroRNNLayer, NeuroRNN

# Set device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- RNN ---------
# model-related
input_size = n_in # features
sequence_length = 50 # timesteps
hidden_size = 200 # RNN units
num_layers = 1 # RNN layers
output_dim = 1
nonlin='relu'
alpha_r = 1
alpha_s = 1

# training-related
num_epochs = 10000
learning_rate = .0005
batch_size = 50
clip_value = .5
lambda_omega = 0
lambda_l2 = .0001
le_init = True # orhan_ma init technique 
alpha_r = 1
alpha_s = 1
diag_val = 1.0
offdiag_val = .2


# other arguments
load_model = False
model_name = 'det_trial'

try:
    os.mkdir('/Users/ebrahimfeghhi/NCEL/rnn_modeling/saved_models/' + model_name)
except:
    print("Folder may have been created") 
   
# init. model
rnnLayer = NeuroRNNLayer(NeuroRNNCell, input_size, hidden_size, alpha_r, alpha_s, nonlin)
model = NeuroRNN(rnnLayer, output_dim, ratio=.8)

# init. hidden state with zeros 
RNNState = namedtuple('RNNState', ['r', 'I'])
state = RNNState(torch.zeros(hidden_size, batch_size), torch.zeros(hidden_size, batch_size))

# init. Wrec appropriately 
if dale and le_init:
    model.LeInitDale(diag_val, offdiag_val)
    
elif dale and not le_init: 
    model.dale_weight_init()

else:
    model.LeInit(diag_val, offdiag_val) 

if load_model:
    model = NeuroRNN(rnnLayer, output_dim)
    model.load_state_dict(torch.load('saved_models'))             


optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)

trainloader = DataLoader(training_set, batch_size=batch_size,
        shuffle=True)

first_run=True

# print model information
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

for epoch in range(num_epochs):
    
    loop = tqdm(trainloader)
    
    for i, (data, targets) in enumerate(loop):
        
        # load data to device
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        if pascanu:
            omega = pascanu_regularizer(model, nonlin, first_run)
        else:
            omega = 0
    
        # forward
        preds, _ = model(data, state)
        loss = criterion(preds, targets)
        
        # backward
        optimizer.zero_grad()
        
        total_loss = loss + omega
        
        total_loss.backward()
        
        # gradient step 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        # enforce Dale
        if dale:
            model.enforce_dale()
            
        training_set.performance(data, targets, preds)
            
        # update progress bar
        loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
        
        first_run = False
        
        
# ---- Save important information and model dict ----
base_file = 'saved_models/' + model_name + '/' 

model_info = {'input_size': input_size, 'hidden_size': hidden_size, 'alpha_r':alpha_r, 'alpha_s':alpha_s, 'nonlin':nonlin
, 'diag_val: ': diag_val, 'offdiag_val: ': offdiag_val, 'batch_size': batch_size, 'lr':learning_rate, 'clip_value':clip_value, 'lambda_l2':lambda_l2, 'num_epochs':num_epochs, 'le_init':le_init, 'dale':dale, 'pascanu':pascanu, 'output_dim':output_dim, 'data_code': data_code}

mi_file = base_file + 'model_info_dict.json'                                   
with open (mi_file, 'w') as f:
    json.dump(model_info, f) 

torch.save(model.state_dict(), base_file + model_name + '.pth')
