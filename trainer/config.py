import torch
import torch.utils.data
import helpers


class TrainConfig:
    def __init__(self, train_set, validation_set, batch_size, epochs, lr, momentum, criterion, seed=None):
        helpers.set_seed(seed)
        self.train_set_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )
        self.validation_set_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=True
        )
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.criterion = criterion
