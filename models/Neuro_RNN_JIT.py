import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
import warnings
from typing import List, Tuple
from torch import Tensor
import numbers
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
JIT implementation provides substantial speed-up and reduction in necessary computational resources.
Allows NeuroRNN to be run on CPU.
'''

# computes a single RNN hidden state update
class NeuroRNNCell(jit.ScriptModule):
    
    def __init__(self, input_size, hidden_size, alpha_r, alpha_s, nonlinearity, bias=False):
        
        super(NeuroRNNCell, self).__init__()
        self.input_size = input_size
        self.nonlinearity = nonlinearity
        self.hidden_size = hidden_size 
        self.alpha_r = alpha_r
        self.alpha_s = alpha_s
        self.bias = bias 
        self.Win = Parameter(torch.Tensor(hidden_size, input_size))
        self.Wrec = Parameter(torch.Tensor(hidden_size, hidden_size))
        
        if bias:
            self.bin = Parameter(torch.Tensor(hidden_size, hidden_size))
            self.brec = Parameter(torch.Tensor(hidden_size))
            
        # init weights 
        nn.init.xavier_normal_(self.Win, .95)
        nn.init.orthogonal_(self.Wrec)
        
    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        r, I = state 
       
        I = (1-self.alpha_s)*I + self.alpha_s*(torch.mm(self.Wrec, r) + torch.mm(self.Win, input.T))
        
        if self.nonlinearity=='tanh':
            r = (1-self.alpha_r)*r + self.alpha_r*(torch.tanh(I) + 1)
        else:
            r = (1-self.alpha_r)*r + self.alpha_r*(torch.nn.functional.relu(I))
        
        return r, (r, I) 

# loops through input to update RNN hidden state
class NeuroRNNLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(NeuroRNNLayer, self).__init__()
        self.cell = cell(*cell_args)
        
        
    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i].float(), state)
            outputs += [out]
            
        return torch.stack(outputs).permute(2,1,0), state

# defines RNN model with biological features (Dale's Principle for now)
class NeuroRNN(jit.ScriptModule):
    
    def __init__(self, rnn_layer, output_dim, ratio): 
        super(NeuroRNN, self).__init__()
        self.rnn_layer = rnn_layer
        self.hidden_size = rnn_layer.cell.hidden_size
        self.Wout = Parameter(torch.Tensor(output_dim, self.hidden_size))
        self.ratio = ratio
        
        nn.init.xavier_normal_(self.Wout, .95)
        
    def dale_weight_init(self):

        with torch.no_grad():
            num_exc = np.int(self.ratio*self.hidden_size)
            num_inh = np.int(self.hidden_size - num_exc)

            D = torch.diag_embed(torch.cat((torch.ones(num_exc), -1*torch.ones(num_inh)))) 
            self.rnn_layer.cell.Wrec = Parameter(torch.abs(self.rnn_layer.cell.Wrec.detach()).matmul(D))
            
    # Written by Emin Orhan       
    def LeInit(self, diag_val=1.0, offdiag_val=.2):
        
        shape = self.hidden_size
        var = 1/(shape**.5)
        std = var**.5

        off_diag_part = offdiag_val * torch.normal(0, std, size=(shape, shape))

        with torch.no_grad():

            self.rnn_layer.cell.Wrec = Parameter(torch.eye(shape) * diag_val + off_diag_part - torch.diag(torch.diag(off_diag_part)))
            
    # LeInit + Dale Init.         
    def LeInitDale(self, diag_val=1.0, offdiag_val=.2):
        
        shape = self.hidden_size
        var = 1/(shape**.5)
        std = var**.5

        num_exc = np.int(self.ratio*shape)
        num_inh = np.int(shape - num_exc)
        
        # sample from two normal distribution for exc. and inh. neurons, mean is set so that mean of Wrec equals 0 
        # to follow Xavier. Init. guidelines
        off_diag_part_exc = offdiag_val * torch.normal(2*std, std, size=(shape, num_exc))
        off_diag_part_inh = offdiag_val * torch.normal(-2*std * 1./(1-self.ratio), std, size=(shape, num_inh))
        
        # set elements which cross zero in normal dist. to zero
        off_diag_part_exc.clamp(min=0)
        off_diag_part_inh.clamp(max=0)
        
        off_diag_part = torch.cat((off_diag_part_exc.T, off_diag_part_inh.T)).T
        
        D = torch.diag_embed(torch.cat((torch.ones(num_exc), -1*torch.ones(num_inh))))

        with torch.no_grad():

            self.rnn_layer.cell.Wrec = Parameter(D * diag_val + off_diag_part - torch.diag(torch.diag(off_diag_part)))

    def enforce_dale(self):


        num_exc = np.int(self.ratio*self.hidden_size)
        num_inh = np.int(self.hidden_size - num_exc)

        self.rnn_layer.cell.Wrec[:num_exc, :].clamp(min=0)
        self.rnn_layer.cell.Wrec[num_exc:, :].clamp(max=0)

    
    def forward(self, input, state):
        
        out, state = self.rnn_layer(input, state) 
        out1 = torch.matmul(self.Wout, out).permute(0,2,1)
        return out1, out
        
        
    