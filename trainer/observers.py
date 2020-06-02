class TrainingObserver:
    def update(self, network, epoch, iteration, loss):
        pass

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        pass


class EmptyObserver(TrainingObserver):
    def __init__(self):
        pass

    def update(self, network, epoch, iteration, loss):
        pass

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        pass


class DummyPrintObserver(TrainingObserver):
    def __init__(self):
        pass

    def update(self, network, epoch, iteration, loss):
        if not iteration % 50:
            print(f'epoch - {epoch}, iteration - {iteration}, loss - {loss}')

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        print(f'Epoch {epoch} validation: loss - {validation_loss}, accuracy - {validation_accuracy}')
