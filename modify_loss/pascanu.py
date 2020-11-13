import torch 
import numpy as np

'''
Implementation of regularization term to prevent vanishing gradients in RNN.
Introduced by Pascanu et. al (2013)
'''

def pascanu_regularizer(model, nonlin, first_run):
    
    '''
    
     Step 1: compute gradient of ht w.r.t ht-1
         This equals Wrec.T*I(Winxt + Wrecht-1 + b), I(x) = 0 if x <= 0, and 1 else
         Compute I(Winxt + Wrecht-1 + b) from ht

     Step 2: multiply BPTT computed dloss/dht w/ dht/dht-1 
     
     Step 3: calculate omega using Pascanu's formula 
     
    '''
    
    if first_run:
        return 0
    
    if nonlin == 'relu':
        
        omega = torch.zeros(1)
        omega.requries_grad=True
        batch_size = model.h_t.grad.shape[0]
       
        # Step 1 
        h_t_binary = torch.diag_embed(torch.squeeze(model.h_t.grad.detach())) # (batch_size, sequence_len, hidden_size, hidden_size) 
        h_t_binary[h_t_binary!=0] = 1 # convert to I(...) 
        dht_dht_prev = torch.matmul(model.rnn_layer.cell.Wrec.T, h_t_binary) # (batch_size, sequence_len, hidden_size, hidden_size) 

        # Step 2 
        dl_dht_prev = torch.squeeze(torch.matmul(torch.unsqueeze(model.h_t.grad.detach(),axis=2), dht_dht_prev)) # (batch_size, seq_len,  hidden_size)
                                    
        # Step 3   
        omega = torch.sum(torch.sum(torch.pow((torch.norm(dl_dht_prev,dim=2) / (torch.norm(torch.squeeze(model.h_t.grad.detach()),dim=2))) - 1, 2),axis=1)) / batch_size
        
        return omega 

    if nonlin == 'tanh':
        
        '''
        Similiar steps to 
        
        omega = torch.zeros(1)
        omega.requries_grad=True
        batch_size = model.h_t.grad.shape[0]
        
        
        print("Code has not been written for other nonlin functions yet, omega is 0")
        '''
        
        omega = 0 
        
    return omega