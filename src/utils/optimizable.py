# pragma: no cover
import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class Optimizable(ABC):
    """
    Abstract class defined a function that can be optimized
    """

    def __init__(self, parameters: np.ndarray) -> None:
        self.parameters = parameters

    def set_params(self, parameters: np.ndarray) -> None:
        self.parameters = parameters

    def get_params(self) -> np.ndarray:
        return self.parameters

    @abstractmethod
    def gradient(self) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self) -> Any:
        pass
