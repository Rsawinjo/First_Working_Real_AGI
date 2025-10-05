from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for LLM modules."""
    @abstractmethod
    def generate(self, prompt, context=None):
        pass
    @abstractmethod
    def get_status(self):
        pass
