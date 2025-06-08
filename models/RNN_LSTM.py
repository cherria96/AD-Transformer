import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_wo_pos
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.c_out = configs.c_out
        self.revin = configs.revin
        self.pred_len = configs.pred_len

        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.rnn = nn.RNN(configs.d_model, configs.d_model, configs.e_layers, batch_first=True, dropout = configs.dropout)
        self.fc1 = nn.Linear(configs.d_model, configs.c_out)
        self.configs = configs
        
    def forward(self, x, x_mark, iterative = False):
        # x = x[:,:, 1:-1]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        if self.revin:
            x = self.revin_layer(x, 'norm') 

        x = self.embedding(x, x_mark)
        if not iterative:
            out, hn = self.rnn(x, h0)  # out: shape(batch, seq_len, hidden_size)
            out = self.fc1(out[:, -self.pred_len:])  # Predict only the last `pred_len` steps

        # Iterative prediction
        else:
            outputs = []
            input_seq = x
            for step in range(self.pred_len):
                out, h0 = self.rnn(input_seq, h0)  # Use RNN with the current input and hidden state
                step_output = self.fc1(out[:, -1:, :])  # Predict one step ahead
                outputs.append(step_output)

                # Prepare the next input by appending the predicted step
                step_output_embedded = self.embedding(step_output, x_mark[:, step:step+1, :])
                input_seq = torch.cat((x[:, 1:, :], step_output_embedded), dim=1)

            out = torch.cat(outputs, dim=1)  # Combine all iterative outputs
        # out = out.permute(0,2,1)
        # out = self.fc2(out).permute(0,2,1)
        if self.revin:
            out = self.revin_layer(out, 'denorm') 

        return out

