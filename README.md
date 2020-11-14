# neuroRNN
This respository is intended to serve as a flexible framework for training biologically plausible RNNs.
I have currently implemented Dale's principle and adaptive time constants, and I aim to provide other features 
such as reciprocal connections between layers and spiking neurons in the future.

To access the most up to date model, go to models/Neuro_RNN_JIT.py. 

All custom datasets can be found in the datasets folder and are written according to the Pytorch dataset class outline. 

Modifications to the loss function, including Pascanu's regularization, are in the modify_loss folder. 

Functions to analyze the inner workings of a trained RNN are in the activity_analysis folder. Currently there is only
the sequential activity metric written by Emin Orhan from Orhan and Ma, Nat. Neuro (2013). 

To train an RNN modify and run rnn_script.py.  

