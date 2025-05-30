import numpy as np
from utils.optimizable import Optimizable
from utils.visualizer import OptimizationVisualizer
from manim import *
from typing import Optional, Any, Tuple

class GradientDescent:
    '''
    Generic gradient descent class
    Takes as an input learning rate, max_iterations, and an Optimizable object with parameters (getters, setters), gradient, forward and 
    '''
    def __init__(
        self, 
        learning_rate: float, 
        max_iterations: int, 
        optimizable: Optimizable, 
        tolerance: Optional[float] = None, 
        visualizer: Optional[OptimizationVisualizer] = None
    ) -> None:
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.optimizable = optimizable
        self.tolerance = tolerance
        self.visualizer = visualizer
        
            


    def find_optimal(self) -> Tuple[np.ndarray, Any]:
        """
        Optimize the optimizable function passed and visualize it if visualizer defined
        """
        if self.visualizer:
            param_history = []
            value_history = []
            gradient_history = []
        
        for i in range(self.max_iterations):
            
            old_params =  self.optimizable.get_params()            
            gradient = self.optimizable.gradient(old_params)

            if self.visualizer:
                value = self.optimizable.forward(old_params)
                param_history.append(old_params.copy())
                value_history.append(value)
                gradient_history.append(gradient.copy())

            new_params = old_params - self.learning_rate * gradient
            self.optimizable.set_params(new_params)

            if self.tolerance:
                if np.linalg.norm(gradient) < self.tolerance:
                    break

            if np.linalg.norm(gradient) == 0:
                break

        if self.visualizer:
            self.visualizer.visualize(param_history, value_history, gradient_history)

        return self.optimizable.get_params(), self.optimizable.forward(self.optimizable.get_params())
