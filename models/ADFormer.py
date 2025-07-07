import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
from layers.RevIN import RevIN
import numpy as np
from utils.timeseries import series_decomp


class Model(nn.Module): 
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.series_decomposition = configs.series_decomposition
        self.kernel_size = configs.kernel_size
        self.decompsition = series_decomp(self.kernel_size)
        self.revin = configs.revin
        self.wodenorm = configs.wodenorm
        if self.revin: self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, patch_len = configs.patch_len)
            self.dec_embedding = DataEmbedding(configs.dec_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, patch_len = configs.patch_len)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
            self.dec_embedding = DataEmbedding(configs.dec_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in * configs.patch_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, patch_len = configs.patch_len)
        # Patching
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch
        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 1)
        if self.padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
            self.patch_num += 1

        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out * configs.patch_len, bias=True)
        )
        self.final_projection = nn.Sequential(
                                nn.Linear(configs.label_len + configs.pred_len, configs.label_len + configs.pred_len),
                                nn.ReLU(),  # Add ReLU activation (optional, can be replaced or removed)
                                nn.Dropout(configs.dropout),  # Optional dropout for regularization
                                nn.Linear(configs.label_len + configs.pred_len, configs.label_len + configs.pred_len),
                                )
    def _patching(self, x):
        x = x.permute(0,2,1)
        self.original_seq_len = x.size(-1)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
            self.padded_seq_len = x.size(-1)
        else:
            self.padded_seq_len=self.original_seq_len
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # x: [bs x enc_in x patch_num x patch_len]
        x = x.permute(0,2,3,1)                                                              # x: [bs x patch_num x patch_len x enc_in]
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))      # x: [bs x patch_num x patch_len * enc_in]
        return x
    def _unpatching(self, x):
        # x (b, c_out, patch_num, patch_len)
        b, c_out, pn, pl = x.shape
        x = x.permute(0, 1, 3, 2).reshape(b * c_out, pl, pn)
        folded = F.fold(
            x,
            output_size = (1, self.padded_seq_len),
            kernel_size = (1, self.patch_len),
            stride = (1, self.stride),
        )
        folded = folded.view(b, c_out, -1)
        folded = folded[:, :, :self.original_seq_len]
        return folded
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: (b, context, enc_in)
        if self.series_decomposition: 
            seasonal_init, trend_init = self.decompsition(x_enc)   
            x_enc = seasonal_init + trend_init        
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm') 
            
        
        # do patching
        x_enc = self._patching(x_enc)
        x_mark_enc = self._patching(x_mark_enc)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # (bs x patch_num x d_model)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        x_dec = self._patching(x_dec)
        x_mark_dec = self._patching(x_mark_dec)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # unpatching
        # dec_out: (b, patch_num, patch_len * c_out) 
        dec_out = dec_out.reshape(dec_out.shape[0], dec_out.shape[1], self.patch_len, -1)     # x: [bs x patch_num x patch_len * enc_in]
        dec_out = dec_out.permute(0, 3, 1, 2)  # [B, c_out, patch_num, patch_len]
        dec_out = self._unpatching(dec_out) # (b, c_out, label+pred_len)
        
        dec_out = self.final_projection(dec_out)
        dec_out = dec_out.permute(0,2,1)
        
        if self.revin and not self.wodenorm:
            dec_out = self.revin_layer(dec_out, 'denorm')
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]