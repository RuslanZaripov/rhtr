from abc import abstractmethod, ABC
from typing import List


class Segmentor(ABC):
    @abstractmethod
    def predict(self, images) -> dict:
        """ """
        pass


class Recognizer(ABC):
    @abstractmethod
    def predict(self, images) -> List[str]:
        pass
