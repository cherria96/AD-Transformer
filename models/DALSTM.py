import torch
import torch.nn as nn
from layers.LSTM_EncDec import Encoder, Decoder
from layers.Embed import DataEmbedding_wo_pos
    
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.encoder = Encoder(configs.d_model, configs.d_model, configs.e_layers, configs.dropout)
        self.decoder = Decoder(configs.d_model, configs.c_out, configs.d_layers, configs.dropout)      
        self.pred_len = configs.pred_len
    def forward(self, x, x_mark):
        x = self.embedding(x, x_mark)
        encoder_outputs, h, c = self.encoder(x)
        predictions = []
        
        for _ in range(self.pred_len):
            prediction, (h, c) = self.decoder(encoder_outputs, h, c)
            predictions.append(prediction.unsqueeze(1))
                
        predictions = torch.cat(predictions, dim=1)  # (batch, pred_len, c_out)
        return predictions