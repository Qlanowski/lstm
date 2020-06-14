import pandas as pd


class TrainingObserver:
    def update(self, network, epoch, iteration, loss):
        pass

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        pass


class DummyPrintObserver(TrainingObserver):
    def update(self, network, epoch, iteration, loss):
        if not iteration % 50:
            print('epoch {:2d}, iteration {:4d}, loss {:6.4f}'.format(epoch, iteration, loss))

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        print('epoch {:2d} validation: loss {:.4f}, accuracy - {:.4f}'.format(epoch, validation_loss, validation_accuracy))


class CollectObserver(DummyPrintObserver):
    def __init__(self):
        self.data = pd.DataFrame(columns=["accuracy", "loss"])
        self.data.index.name = "epoch"

    def update(self, network, epoch, iteration, loss):
        super().update(network, epoch, iteration, loss)

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        super().validation_update(network, epoch, validation_accuracy, validation_loss)
        self.data.loc[epoch] = [validation_accuracy, validation_loss]

    def get_results(self):
        return self.data.copy()