import torch
from torch import nn
class LSTM_Network(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, classes, **lstm_kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, **lstm_kwargs)
        self.linear = nn.Linear(hidden_dim, classes)
        
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])