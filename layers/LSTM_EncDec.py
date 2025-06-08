import torch
import torch.nn as nn

class InputAttention(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(InputAttention, self).__init__()
        self.W = nn.Linear(hidden_size, input_size, bias=False)
        self.U = nn.Linear(input_size, input_size, bias=False)
        self.v = nn.Linear(input_size, 1, bias=False)

    def forward(self, encoder_hidden, inputs):
        # encoder_hidden: (batch, hidden_size)
        # inputs: (batch, seq_len, input_size)
        h = self.W(encoder_hidden).unsqueeze(1)  # (batch, 1, input_size)
        x = self.U(inputs)  # (batch, seq_len, input_size)
        
        scores = self.v(torch.tanh(h + x))  # (batch, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        context = (attention_weights * inputs).sum(dim=1)  # (batch, input_size)
        return context, attention_weights.squeeze(-1)
    
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size)
        # encoder_outputs: (batch, seq_len, hidden_size)
        h = self.W(decoder_hidden).unsqueeze(1)  # (batch, 1, hidden_size)
        x = self.U(encoder_outputs)  # (batch, seq_len, hidden_size)
        
        scores = self.v(torch.tanh(h + x))  # (batch, seq_len, 1)
        attention_weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        context = (attention_weights * encoder_outputs).sum(dim=1)  # (batch, hidden_size)
        return context, attention_weights.squeeze(-1)
    
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.input_attention = InputAttention(hidden_size, input_size)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        encoder_outputs = []
        h, c = self.init_hidden(batch_size, x.device)
        
        for t in range(seq_len):
            context, _ = self.input_attention(h[-1], x)
            _, (h, c) = self.lstm(context.unsqueeze(1), (h, c))
            encoder_outputs.append(h[-1].unsqueeze(1))
        
        encoder_outputs = torch.cat(encoder_outputs, dim=1)  # (batch, seq_len, hidden_size)
        return encoder_outputs, h, c

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        return h, c
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.temporal_attention = TemporalAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, h, c):
        context, _ = self.temporal_attention(h[-1], encoder_outputs)
        output, (h, c) = self.lstm(context.unsqueeze(1), (h, c))
        prediction = self.fc(output.squeeze(1))  # (batch, output_size)
        return prediction, (h, c)
