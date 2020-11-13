# neuroRNN
This respository is intended to serve as a flexible framework for training biologically plausible RNNs.
Some of the features I aim to provide are Dale's principle, adaptive time constants, and reciprocal connections between layers.
I am currently focusing on implementing Dale's principle and understanding its impact.

To access the most up to date model, go to models/Neuro_RNN_JIT.py. Adaptive time constants and Dale's Principle are currently implemented. 

All custom datasets can be found in the datasets folder, and follow Pytorch dataset class format. 

Modifications to the loss function, including Pascanu's regularization are in the modify_loss folder. 

Functions to analyze the inner workings of a trained RNN are in the activity_analysis folder. Currently only
the sequential activity metric written by Emin Orhan is in the folder. 

