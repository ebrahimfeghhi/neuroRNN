import torch 
import numpy as np


def retrieve(data_code, loss_params):
    
    if data_code==0:
        
        from custom_loss import orhan_ma
        criterion = orhan_ma(**loss_params)
        return criterion
    