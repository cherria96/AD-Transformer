import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.c_out = configs.c_out
        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.lstm = nn.LSTM(configs.d_model, configs.d_model, configs.e_layers, batch_first=True, dropout = configs.dropout)
        self.fc1 = nn.Linear(configs.d_model, configs.c_out)
        self.fc2 = nn.Linear(configs.seq_len, configs.label_len + configs.pred_len)
        self.configs = configs
        
    def forward(self, x, x_mark):
        # x = x[:,:, 1:-1]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        x = self.embedding(x, x_mark)
        out, (hn, cn) = self.lstm(x, (h0, c0)) # out: shape(batch, seq_len, hidden_size)
        
        out = self.fc1(out[:, -self.configs.pred_len:])
        # out = out.permute(0,2,1)
        # out = self.fc2(out).permute(0,2,1)
        return out