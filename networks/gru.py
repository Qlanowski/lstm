import torch
from torch import nn
class GRU_Network(torch.nn.Module) :
    def __init__(self, embedding_dim, hidden_dim, classes, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, classes)
        
    def forward(self, x):
        output, hidden = self.gru(x)
        return self.linear(hidden[-1])