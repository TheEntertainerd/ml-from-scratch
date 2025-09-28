# pragma: no cover
from abc import ABC, abstractmethod
import numpy as np


class OptimizationVisualizer(ABC):
    """
    Abstract class to visualize a function with manim
    """

    @abstractmethod
    def visualize(
        self, param_history: list[np.ndarray], value_history: list[float], gradient_history: list[np.ndarray]
    ) -> None:
        pass
