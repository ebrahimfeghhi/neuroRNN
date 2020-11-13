import sys

def retrieve(data_code, num_trials, n_in):
    
    if data_code == 0:
        
        from Orhan_Ma import DelayedEstimationTask
        params = {'num_trials':num_trials, 'n_in':n_in, 'stim_dur':25, 'delay_dur':10, 'resp_dur':25}
        dataset = DelayedEstimationTask(**params)
        print("Input shape: ", dataset.data.shape)
        return dataset
    
