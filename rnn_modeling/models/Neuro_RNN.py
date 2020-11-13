import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.parameter import Parameter
import torch.jit as jit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuroRNN(nn.Module):
    
    def __init__(self, input_size, output_dim, alpha_r, alpha_s, nonlinearity, hidden_size, bias, ratio):
        
        super(NeuroRNN, self).__init__()
        self.input_size = input_size
        self.nonlinearity = nonlinearity
        self.hidden_size = hidden_size 
        self.alpha_r = alpha_r
        self.alpha_s = alpha_s
        self.bias = bias 
        self.output_dim = output_dim
        self.Win = Parameter(torch.Tensor(hidden_size, input_size))
        self.Win.requires_grad = True 
        self.Wrec = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.Wrec.requires_grad = True 
        self.Wout = Parameter(torch.Tensor(output_dim, hidden_size))
        self.Wout.requires_grad = True
        self.ratio = ratio
        
        if bias:
            self.bin = Parameter(torch.Tensor(hidden_size, hidden_size))
            self.brec = Parameter(torch.Tensor(hidden_size))
            
        # init weights 
        nn.init.orthogonal_(self.Win)
        nn.init.orthogonal_(self.Wrec)
        nn.init.uniform_(self.Wout)
        
            
    def nonlin(self, inp):
            
        if self.nonlinearity == 'relu':
            nn.ReLU(inp)
            return inp
        else:
            inp = torch.tanh(inp)
            return inp

    def forward(self, x):
        
        # x shape: (batch_size, seq_len, input_size)
        I = torch.zeros(self.hidden_size, x.size(0)).to(device) # (hidden_size, batch_size)
        r = torch.zeros(self.hidden_size, x.size(0)).to(device)
        out = torch.zeros(x.size(0), x.size(1), self.output_dim).to(device) # (batch_size, seq_len, output_dim) 
        
        # for pascanu regularization 
        #r_total = torch.zeros(x.size(0), self.hidden_size, x.size(1))
        #r_total.requires_grad = True 
        
        for t in range(x.size(1)):
            I = (1-self.alpha_s)*I + self.alpha_s*(torch.mm(self.Wrec,r) + torch.mm(self.Win, x[:, t].T ))
            r = (1-self.alpha_r)*r + self.alpha_r*(self.nonlin(I))
            #r_total[:, :, t] = r.T
            out[:, t, :] = torch.mm(self.Wout, r).T 
            
        del I 
        del r
            
        return out 
    
    def dale_weight_init(self):

        with torch.no_grad():

            num_exc = np.int(self.ratio[0]*self.hidden_size)
            num_inh = np.int(self.hidden_size - num_exc)

            D = torch.diag_embed(torch.cat((torch.ones(num_exc), -1*torch.ones(num_inh)))) 
            self.Wrec = torch.nn.Parameter(torch.abs(self.Wrec.detach()).matmul(D))


    def enforce_dale(self):
        
        with torch.no_grad():
            
            num_exc = np.int(self.ratio[0]*self.hidden_size)
            num_inh = np.int(self.hidden_size - num_exc)

            self.Wrec[:num_exc, :].clamp(min=0)
            self.Wrec[num_exc:, :].clamp(max=0)
            
            