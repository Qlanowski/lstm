import pandas as pd


class TrainingObserver:
    def update(self, network, epoch, iteration, loss):
        pass

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        pass


class DummyPrintObserver(TrainingObserver):
    def update(self, network, epoch, iteration, loss):
        if not iteration % 50:
            print(f'epoch - {epoch}, iteration - {iteration}, loss - {loss}')

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        print(f'Epoch {epoch} validation: loss - {validation_loss}, accuracy - {validation_accuracy}')


class CollectObserver(DummyPrintObserver):
    def __init__(self):
        self.data = pd.DataFrame(columns=["epoch", "accuracy", "loss"])

    def update(self, network, epoch, iteration, loss):
        super().update(network, epoch, iteration, loss)

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        super().validation_update(network, epoch, validation_accuracy, validation_loss)
        self.data.iloc[len(self.data)] = [epoch, validation_accuracy, validation_loss]

    def get_results(self):
        return self.data.copy()