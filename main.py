import sys
import os
import dataloaders.loader
import networks
import trainer
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import random
import numpy as np
import torch
import torch.backends.cudnn


def __set_seed(seed):
    """
        Set seed for all torch func and methods.
        See ref: https://github.com/pytorch/pytorch/issues/7068
        """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main(conf_files):
    data = dataloaders.loader.load_train()

    for conf_file in conf_files:
        with open(conf_file) as conf_json:
            conf = json.load(conf_json)

        __set_seed(conf["seed"])

        train_set, valid_set = dataloaders.loader.split_random(data, frac=conf["split_frac"])
        network = getattr(networks, conf["network_name"])(**conf["network_params"])
        observer = trainer.CollectObserver()

        trainer.train_network(
            network=network,
            criterion=getattr(nn, conf["criterion"])(),
            optimizer=getattr(optim, conf["optim_name"])(
                params=network.parameters(), 
                **conf["optim_params"]
            ),
            epochs=conf["epochs"],
            train_set_loader=DataLoader(
                train_set,
                batch_size=conf["batch_size"],
                shuffle=True
            ),
            validation_set_loader=DataLoader(
                valid_set,
                batch_size=conf["batch_size"],
                shuffle=True
            ),
            observer=observer)

        results_df = observer.get_results()
        accuracy_fig = trainer.create_accuracy_plot(results_df)
        loss_fig = trainer.create_loss_plot(results_df)

        results_dir = os.path.join("results", conf["name"])
        os.makedirs(results_dir, exist_ok=True)

        results_df.to_csv(os.path.join(results_dir, "results.csv"))
        accuracy_fig.savefig(os.path.join(results_dir, "accuracy.png"))
        loss_fig.savefig(os.path.join(results_dir, "loss.png"))
    


if __name__ == '__main__':
    main(sys.argv[1:])
