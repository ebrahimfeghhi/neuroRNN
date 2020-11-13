import torch 
import numpy as np


class orhan_ma():
    
        def __init__(self, resp_dur):
            self.resp_dur = resp_dur
   
        def __call__(self, targets, scores): 
        
            loss = torch.zeros(1)
            loss.requires_grad=True
            loss = torch.mean(torch.remainder(torch.abs_(scores[:,-self.resp_dur:,:] - targets[:,-self.resp_dur:,:]), np.pi))
            return loss
