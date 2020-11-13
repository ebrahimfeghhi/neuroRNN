# Targets First Class
class TF(Dataset):
    
    def __init__(self, num_trials, trial_length, coh, start_delay, target_length, mid_delay, chech_length, post_delay):
        
        self.num_trials = num_trials
        self.trial_length = trial_length
        self.coh = coh
        self.start_delay = start_delay
        self.target_length = target_length
        self.mid_delay = mid_delay
        self.check_length = check_length
        self.post_delay = post_delay
        
        X = np.zeros((self.num_trials, self.trial_length))
        
        # create useful variables
        num_coh = self.num_trials // len(self.coh)
        target_onset = self.start_delay + self.check_length + self.mid_delay
        
        # uniformly assign coherences 
        X[:, self.start_delay:self.start_delay + self.check_length] = np.tile(np.repeat(self.coh, num_coh), (self.check_length,1)).T
        
        # set target location, -1 means red target is on left, 1 means red target is on right
        X[:, target_onset:target_onset+self.target_length] = np.tile([-1,1], (self.target_length, num_trials//2)).T
        
        Y = np.zeros((self.num_trials, self.trial_length, 2))
        
        # green checkerboard trials, 0 is left reach, 1 is right reach
        Y[:self.num_trials//2, target_onset:target_onset+self.target_length, 0] = np.tile([0,1], (self.target_length, num_trials//4)).T
        Y[:self.num_trials//2, target_onset:target_onset+self.target_length, 1] = np.tile([1,0], (self.target_length, num_trials//4)).T
        
        # red checkerboard trials
        Y[self.num_trials//2:, target_onset:target_onset+self.target_length, 0] = np.tile([1,0], (self.target_length, num_trials//4)).T
        Y[self.num_trials//2:, target_onset:target_onset+self.target_length, 1] = np.tile([0,1], (self.target_length, num_trials//4)).T
        
        self.data = torch.from_numpy(np.expand_dims(X,axis=-1)).float()
        self.labels = torch.from_numpy(Y).float()
        
    def __len__(self):
        
        return (self.data.shape[0])
    
    def __getitem__(self, idx):
        
        return self.data[idx], self.labels[idx]