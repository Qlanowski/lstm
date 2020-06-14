import torch
from torch import nn
class GRU_Network(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, classes, **gru_kwargs):
        super().__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, **gru_kwargs)
        self.linear = nn.Linear(hidden_dim, classes)
        
    def forward(self, x):
        output, hidden = self.gru(x)
        return self.linear(hidden[-1])