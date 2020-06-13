import torch
from torch import nn
class LSTM_Network(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, classes)
        
    def forward(self, x):
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])