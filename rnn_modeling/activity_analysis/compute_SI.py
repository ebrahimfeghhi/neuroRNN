# Code written by Emin Orhan 

import numpy as np 
from scipy.stats import entropy 

def compute_SI(hidr, entrpy_bins=5, window_size=5, r_threshold=.1):
    
    # Compute SI scores for a number of trials
    # hidr: activities of recurrent units
    # entrpy_bins: number of bins used for estimating peak time entropy (20 in the paper)
    # window_size: window size around peak time for calculating ridge-to-background ratio (5 in the paper)
    # r_threshold: only consider recurrent units with mean responses above this value (0.1 in the paper)
    
    bs = hidr.shape[0]  # number of trials
    ts = hidr.shape[1]  # number of time points

    SI_trial_vec = np.zeros(bs)
    for b in range(bs):
        hidr_t = hidr[b, :, :]
        selected_indx = np.nonzero(np.mean(hidr_t, axis=0) > r_threshold)[0]
        hidr_t = hidr_t[:, selected_indx]

        peak_times = np.argmax(hidr_t, axis=0)
        end_times = np.clip(peak_times + window_size / 2, 0, ts - 1)
        start_times = np.clip(peak_times - window_size / 2, 0, ts - 1)
        print(start_times)
        entrpy = entropy(np.histogram(peak_times, entrpy_bins)[0] + 0.1 * np.ones(entrpy_bins))

        r2b_ratio = np.zeros(len(selected_indx))
        for nind in range(len(selected_indx)):
            mask = np.zeros(ts)
            print(start_times[nind], end_times[nind])
            mask[start_times[nind]:end_times[nind]] = 1
            this_hidr = hidr_t[:, nind]
            ridge = np.mean(this_hidr[start_times[nind]:end_times[nind]])
            backgr = np.mean(np.ma.MaskedArray(this_hidr, mask))
            r2b_ratio[nind] = np.log(ridge) - np.log(backgr)

        SI_trial_vec[b] = np.nanmean(r2b_ratio) + entrpy

    return np.nanmean(SI_trial_vec)