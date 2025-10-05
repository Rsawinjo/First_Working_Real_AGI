from abc import ABC, abstractmethod

class BaseLearning(ABC):
    """Abstract base class for learning modules."""
    @abstractmethod
    def learn(self, input_data):
        pass
    @abstractmethod
    def evaluate(self):
        pass
