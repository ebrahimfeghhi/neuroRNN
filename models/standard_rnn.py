# -------------Standard RNN Model w/ Dale functions---------------

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_dim, ratio):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_dim = output_dim
        self.ratio = ratio
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity=nonlin, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_size, output_dim, bias=False)
            
    def dale_weight_init(self):
        
        with torch.no_grad():
        
            num_exc = np.int(self.ratio[0]*self.hidden_size)
            num_inh = np.int(self.hidden_size - num_exc)

            D = torch.diag_embed(torch.cat((torch.ones(num_exc), -1*torch.ones(num_inh)))) 
            self.rnn.weight_hh_l0 = torch.nn.Parameter(torch.abs(self.rnn.weight_hh_l0.detach()).matmul(D))
                                                    

    def enforce_dale(self):
        
        with torch.no_grad():
            
            num_exc = np.int(self.ratio[0]*self.hidden_size)
            num_inh = np.int(self.hidden_size - num_exc)

            self.rnn.weight_hh_l0[:num_exc, :].clamp(min=0)
            self.rnn.weight_hh_l0[num_exc:, :].clamp(max=0)
              
            
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        h_t, _ = self.rnn(x,h0)
        h_t.retain_grad()
        self.h_t = h_t
        out = self.fc(h_t)
        return out
    
