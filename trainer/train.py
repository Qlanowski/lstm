import torch
import trainer.observers as observers
import copy
import time
import pandas as pd


def __get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_network(
    network,
    criterion,
    optimizer,
    epochs,
    train_set_loader,
    validation_set_loader,
    observer=observers.TrainingObserver()
    ):
    device = __get_device()
    network.to(device)
    network.cuda()

    min_loss = float('inf')
    min_loss_network = None

    for epoch in range(epochs):

        start = time.time()
        for iteration, data in enumerate(train_set_loader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            observer.update(network, epoch, iteration, loss.item())

        validation_loss, validation_accuracy = get_validation_stats(
            network=network,
            validation_set_loader=validation_set_loader,
            criterion=criterion
        )
        observer.validation_update(network, epoch, validation_loss, validation_accuracy)

        end = time.time()
        print("epoch finished in {:.4f}s".format(end - start))

        if validation_loss < min_loss:
            min_loss = validation_loss
            min_loss_network = copy.deepcopy(network)

        if validation_loss / min_loss > 1.3:
            return min_loss_network

    return min_loss_network


def get_validation_stats(network, validation_set_loader, criterion):
    device = __get_device()
    network.to(device)
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in validation_set_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            loss_val = criterion(outputs, labels)
            loss += loss_val.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss, correct / total


def get_submission(network, test_set_loader, idx_to_label):
    device = __get_device()
    network.to(device)
    results = {
        "fname": [],
        "label_idx": [],
        "label": []
    }

    with torch.no_grad():
        for data in test_set_loader:
            inputs = data[0].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            results["label_idx"] += predicted.tolist()
            results["fname"] += list(data[2])
            print("{:6d} records predicted".format(len(results["label_idx"])))

    results["label"] = [
        idx_to_label[value] 
        for value in results["label_idx"]
    ]
    results_df = pd.DataFrame(results)
    return results_df.loc[:, ["fname", "label"]]