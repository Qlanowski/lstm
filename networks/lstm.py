import torch
from torch import nn
class LSTM_fixed_len(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, classes):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, classes)
        
    def forward(self, x, l):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])