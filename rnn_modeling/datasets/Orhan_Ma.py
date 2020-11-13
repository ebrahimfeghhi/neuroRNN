# Code written by Emin Orhan 
import numpy as np
import scipy.stats as scistat
import time
from matplotlib import pyplot as plt

class Task(object):

    def __init__(self, num_trials, max_iter=None):
        self.max_iter = max_iter
        self.num_trials = num_trials
        self.num_iter = 0
 
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1) , self.sample()
        else:
            raise StopIteration()

    def sample(self):
        raise NotImplementedError()
        

class DelayedEstimationTask(Task):
    
    def __init__(self, max_iter=None, num_trials=50, n_loc=1, n_in=None, n_out=1, stim_dur=25, delay_dur=100, resp_dur=25, kappa=2.0, spon_rate=0.1, tr_cond='all_gains'):
        super(DelayedEstimationTask, self).__init__(max_iter=max_iter, num_trials=num_trials)
        self.n_in      = n_in                  # number of neurons per location
        self.n_out     = n_out
        self.n_loc     = n_loc
        self.kappa     = kappa
        self.spon_rate = spon_rate
        self.nneuron   = self.n_in * self.n_loc # total number of input neurons
        self.phi       = np.linspace(0, np.pi, self.n_in)
        self.stim_dur  = stim_dur
        self.delay_dur = delay_dur
        self.resp_dur  = resp_dur
        self.total_dur = stim_dur + delay_dur + resp_dur
        self.tr_cond   = tr_cond
        self.mu_x_list = []
        self.S_list = []
        self.pred_list = []
        self.batch_num=50
        
        # ---- Define Gain -----
        if self.tr_cond == 'all_gains':
            G = (1.0/self.stim_dur) * np.random.choice([1.0], size=(self.n_loc,self.num_trials))
            G = np.repeat(G,self.n_in,axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
        else:
            G = (0.5/self.stim_dur) * np.random.choice([1.0], size=(1,self.num_trials))
            G = np.repeat(G,self.n_in * self.n_loc, axis=0).T
            G = np.tile(G,(self.stim_dur,1,1))
            G = np.swapaxes(G,0,1)
            
        # ---- Uniformly sample mean between 0 and pi ----        
        S1             = np.pi * np.random.rand(self.n_loc, self.num_trials)
        S              = S1.T.copy()
        S1             = np.repeat(S1,self.n_in,axis=0).T
        S1             = np.tile(S1,(self.stim_dur,1,1))
        S1             = np.swapaxes(S1,0,1)
                
        # ---- Lambda values ---- 
        L1             = G * np.exp( self.kappa * (np.cos( 2.0 * (S1 - np.tile(self.phi, (self.num_trials,self.stim_dur,self.n_loc) ) ) ) - 1.0) ) # stim 
        Ld             = (self.spon_rate / self.delay_dur) * np.ones((self.num_trials,self.delay_dur,self.nneuron)) # delay
        Lr             = (self.spon_rate / self.resp_dur) * np.ones((self.num_trials,self.resp_dur,self.nneuron)) # response
    
        # ---- Sample from Poisson with lambda values ----
        R1             = np.random.poisson(L1)
        Rd             = np.random.poisson(Ld)
        Rr             = np.random.poisson(Lr)

        # ---- input is poisson firing rates ----
        example_input  = np.concatenate((R1,Rd,Rr), axis=1)
        
        # ---- output is mean values -----
        example_output = np.repeat(S[:,np.newaxis,:],self.total_dur,axis=1)    
        
        self.data  = example_input
        self.labels = example_output
        
        
    def performance(self, batch_input, batch_output, pred):
        
         # ---- reconstruct original mean value ----
        R1 = batch_input[:, :self.stim_dur, :].numpy()
        cum_R1         = np.sum(R1,axis=1)         
        mu_x           = np.asarray([ np.arctan2( np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.sin(2.0*self.phi)), np.dot(cum_R1[:,i*self.n_in:(i+1)*self.n_in],np.cos(2.0*self.phi))) for i in range(self.n_loc) ]) / 2.0
        mu_x           = (mu_x > 0.0) * mu_x + (mu_x<0.0) * (mu_x + np.pi) 
        mu_x           = mu_x.T
        
        S = batch_output.numpy()
        
        self.mu_x_list.append(mu_x)
        self.S_list.append(S)
        self.pred_list.append(pred.detach().numpy())
        
        self.batch_num += 1
        
        if self.batch_num % 200 == 0:
            s_vec = np.asarray(self.S_list)
            s_red_vec = s_vec[:, :, 0]
            opt_vec = np.asarray(self.mu_x_list)
            net_vec = np.asarray(self.pred_list)

            rmse_opt = np.nanmean(np.mod(np.abs(np.squeeze(s_red_vec) - np.squeeze(opt_vec)), np.pi)) 
            rmse_net = np.nanmean(np.mod(np.abs(np.squeeze(s_vec) - np.squeeze(net_vec)), np.pi))
            infloss  = (rmse_net - rmse_opt) / rmse_opt
            
            self.mu_x_list = []
            self.S_list = [] 
            self.pred_list = []
            
            print("Performance is : ", infloss) 
           
    def __len__(self):
        
        return (self.data.shape[0])
    
    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx]
        
        

        
        
        