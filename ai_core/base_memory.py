from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """Abstract base class for memory modules."""
    @abstractmethod
    def save(self, data):
        pass
    @abstractmethod
    def load(self):
        pass
    @abstractmethod
    def clear(self):
        pass
