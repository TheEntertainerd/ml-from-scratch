from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any, Tuple, List, Dict

class OptimizationVisualizer(ABC):
    """
    Abstract class to visualize a function with manim
    """
    @abstractmethod
    def visualize(
        self, 
        param_history: List[np.ndarray], 
        value_history: List[float], 
        gradient_history: List[np.ndarray]
    ) -> None:
        pass