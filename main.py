import sys
import dataloaders.loader
from networks.lstm import LSTM_fixed_len
import trainer
import torch.nn as nn



def main(config_paths):
    data = dataloaders.loader.load_train()
    print(data.classes)
    network = LSTM_fixed_len(embedding_dim=30, hidden_dim=30, classes=30)
    train_set, valid_set = dataloaders.loader.split_random(data, frac=0.7)

    config = trainer.TrainConfig(train_set=train_set, validation_set=valid_set, batch_size=16, epochs=10, lr=0.01, momentum=0.01, criterion=nn.CrossEntropyLoss(), seed=None)

    trainer.train_network(network=network, config=config, observer=trainer.observers.DummyPrintObserver())
    


if __name__ == '__main__':
    main(sys.argv[1:])
