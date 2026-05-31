from abc import ABC, abstractmethod

class BaseEngine(ABC):
    @abstractmethod
    def train_one_epoch(self, model, loader, optimizer, args, current_epoch=None, total_epochs=None):
        pass

    @abstractmethod
    def evaluate(self, model, loader, args):
        pass
