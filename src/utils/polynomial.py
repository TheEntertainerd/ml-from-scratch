import numpy as np
from utils.optimizable import Optimizable
from utils.visualizer import OptimizationVisualizer
from manim import *
from typing import Optional, Any, Tuple, List, Dict


class MultivariatePolynomial:
    """
    Class representing a multivariate polynomial function and allowing getting the value and derivative at any point
    Polynomials are represented with dictionary tuples/coefficient.
    For example f x,y,x = x^2 + 2xy + 3z^2 would be
    {
    (2,0,0)=1
    (1,1,0)=2
    (0,0,2)=3
    }
    """

    def __init__(self, dictionary_coefficients: Dict[Tuple[int, ...], float]) -> None:

        if len(dictionary_coefficients)==0:
            raise ValueError("Dictionary is empty")
        first_key=next(iter(dictionary_coefficients))
        if not isinstance(first_key,tuple):
            raise ValueError('Key must be tuple')
        
        number_variables=len(first_key)

        for key in dictionary_coefficients:
            if not isinstance(key,tuple):
                raise ValueError('Key must be tuple')
            if len(key)!=number_variables:
                raise ValueError(f'All tuples must have same length')
        self.dictionary_coefficients=dictionary_coefficients


    def __call__(self, values_variables: List[float]) -> float:
        """
        For example to evaluate f x,y,x = x^2 + 2xy + 3z^2  evaluated at (2,3,1) would be 24 
        {
        (2,0,0)=1
        (1,1,0)=2
        (0,0,2)=3
        }
        """        
        
        if len(values_variables) != len(next(iter(self.dictionary_coefficients))):
            print('Number of variables passed does not match that of the polynome')

        result = 0
        for term in self.dictionary_coefficients:
            inter_value= self.dictionary_coefficients[term]
            for exponent, value  in zip(term,values_variables):
                inter_value *= value ** exponent
            result += inter_value
        return result


    def get_derivative(self, values_variables: List[float], index_variable: int) -> float:
        """
        Compute derivative of polynome at parameter passed, for the variable at this index 
        """        
        
        if len(values_variables) != len(next(iter(self.dictionary_coefficients))):
            raise ValueError(f'Number of variables passed does not match that of the polynome')

        result = 0
        for term in self.dictionary_coefficients:
            if term[index_variable]==0:
                continue

            inter_value= self.dictionary_coefficients[term]

            for var, (exponent, value)  in enumerate(zip(term,values_variables)):
                if var == index_variable:
                    inter_value *= exponent * (value ** (exponent - 1))
                else:
                    inter_value *= value ** exponent
            result += inter_value
        return result
    

    def get_gradient(self, values_variables: List[float]) -> np.ndarray:
        """
        Compute gradient of polynome at given point
        """        
        L = []
        for i in range(len(values_variables)):
            L.append(self.get_derivative(values_variables, i))
        
        return np.array(L)



class MultivariatePolynomialOptimizable(Optimizable):
    """
    Class representing the Optimizable polynome adapted for Gradient Descent
    """
    def __init__(self, dictionary_coefficients: Dict[Tuple[int, ...], float], parameters: np.ndarray) -> None:
        super().__init__(parameters)
        self.dictionary_coefficients = dictionary_coefficients
        self.polynome = MultivariatePolynomial(dictionary_coefficients)

    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        return self.polynome.get_gradient(parameters)
    
    def forward(self, parameters: np.ndarray) -> float:
        return self.polynome(parameters)
 
 

class MultivariatePolynomialVisualizer(OptimizationVisualizer):
    def __init__(
        self, 
        optimizable: Optimizable, 
        learning_rate: float, 
        x_range: Tuple[float, float] = (-3, 3), 
        y_range: Optional[Tuple[float, float]] = None, 
        z_range: Optional[Tuple[float, float]] = None, 
        x_step: float = 1.0, 
        y_step: Optional[float] = None, 
        z_step: Optional[float] = None, 
        quality: str = "low_quality"
    ) -> None:
        """
        Class to visualize polynome optimization for Gradient Descent minimization
        """
        self.optimizable = optimizable
        self.learning_rate = learning_rate
        self.x_range = x_range
        self.y_range = y_range if y_range else x_range
        self.z_range = z_range if z_range else x_range
        self.x_step = x_step
        self.y_step = y_step if y_step else x_step
        self.z_step = z_step if z_step else x_step
        self.quality = quality
        
    def _format_number(self, num: float) -> str:
        """Helper function to format numbers up to 3 decimal places"""
        if abs(num) < 1e-10:  # Handle very small numbers as zero
            return "0"
        if num == int(num):
            return str(int(num))
        else:
            formatted = f"{num:.3f}".rstrip('0').rstrip('.')
            return formatted if '.' in formatted else str(int(num))
    
    def _get_plot_dimension(self) -> int:
        """Determine if this is 2D or 3D based on parameter dimension"""
        sample_params = self.optimizable.get_params()
        return len(sample_params)
    
        
    def visualize(
        self, 
        param_history: List[np.ndarray], 
        value_history: List[float], 
        gradient_history: List[np.ndarray]
    ) -> None:
        """Main visualization method"""
        plot_dim = self._get_plot_dimension()
        
        if plot_dim == 1:
            self._visualize_2d(param_history, value_history, gradient_history)
        elif plot_dim == 2:
            self._visualize_3d(param_history, value_history, gradient_history)
        else:
            raise ValueError(f"Cannot visualize optimization for {plot_dim}D parameter space")
    
    def _visualize_2d(
        self, 
        param_history: List[np.ndarray], 
        value_history: List[float], 
        gradient_history: List[np.ndarray]
    ) -> None:
        """2D visualization for single-parameter optimization"""
        
        class GradientDescent2DAnimation(Scene):
            def construct(inner_self):

                alpha_text = Text(f"α = {self._format_number(self.learning_rate)}", 
                                font_size=24, color=WHITE)
                alpha_text.to_corner(UP + RIGHT)
                inner_self.add(alpha_text)

                # Get plot function
                plot_function = lambda x: self.optimizable.forward(np.array([x]))
                
                # If y_range is passed, use it, otherwise, infer it
                if hasattr(self, 'y_range') and self.y_range and len(self.y_range) == 2:
                    plot_y_range = self.y_range
                else:
                    x_vals = np.linspace(self.x_range[0], self.x_range[1], 100)
                    y_vals = [plot_function(x) for x in x_vals]
                    y_min, y_max = min(y_vals), max(y_vals)
                    y_padding = max(0.1, (y_max - y_min) * 0.1)
                    plot_y_range = (y_min - y_padding, y_max + y_padding)
                
                # Ensure y_range is valid
                if abs(plot_y_range[1] - plot_y_range[0]) < 1e-10:
                    center = (plot_y_range[0] + plot_y_range[1]) / 2
                    plot_y_range = (center - 1, center + 1)
                
                # If y_step is passed, use it, otherwise, infer it
                y_step_to_use = self.y_step
                if not y_step_to_use:
                    y_range_size = abs(plot_y_range[1] - plot_y_range[0])
                    y_step_to_use = max(0.1, y_range_size / 5)
                
                # Create axes positioned on the left side
                axes = Axes(
                    x_range=[self.x_range[0], self.x_range[1], self.x_step],
                    y_range=[plot_y_range[0], plot_y_range[1], y_step_to_use],
                    x_length=5,
                    y_length=4,
                    axis_config={"color": BLUE, "include_numbers": True},
                )
                axes.shift(LEFT * 2.5)
                
                # Plot the function
                function_graph = axes.plot(plot_function, x_range=[self.x_range[0], self.x_range[1]], 
                                         color=RED, stroke_width=3)
                
                # Add axes and curve
                inner_self.add(axes, function_graph)
            
                
                # Show starting point
                initial_point = axes.c2p(param_history[0][0], value_history[0])
                optimization_dot = Dot(initial_point, color=YELLOW, radius=0.08)
                
                # Current iteration display
                current_iteration_display = VGroup()
                
                inner_self.add(optimization_dot)
                
                # Animate through optimization steps
                for i in range(len(param_history) - 1):
                    current_x = param_history[i][0]
                    gradient_val = gradient_history[i][0]
                    next_x = param_history[i + 1][0]
                    
                    # Clear previous iteration display
                    if len(current_iteration_display) > 0:
                        inner_self.remove(current_iteration_display)
                    
                    # Create new iteration display
                    x_current_text = Text(f"x{i} = {self._format_number(current_x)}", 
                                        font_size=32, color=BLUE)
                    x_current_text.move_to(RIGHT * 3.5 + UP * 2)
                    
                    gradient_calc = Text(f"f'(x{i}) = {self._format_number(gradient_val)}", 
                                       font_size=32, color=GREEN)
                    gradient_calc.next_to(x_current_text, DOWN, buff=0.3)
                    
                    update_calc = Text(f"x{i+1} = {self._format_number(current_x)} - α·{self._format_number(gradient_val)}", 
                                     font_size=32, color=YELLOW)
                    update_calc.next_to(gradient_calc, DOWN, buff=0.3)
                    
                    update_final = Text(f"x{i+1} = {self._format_number(next_x)}", 
                                      font_size=32, color=YELLOW)
                    update_final.next_to(update_calc, DOWN, buff=0.3)
                    
                    current_iteration_display = VGroup(x_current_text, gradient_calc, update_calc, update_final)
                    
                    # Show current iteration
                    inner_self.play(Write(current_iteration_display), run_time=0.8)
                    inner_self.wait(1.0)
                    
                    # Update point on graph
                    new_point = axes.c2p(next_x, value_history[i + 1])
                    new_dot = Dot(new_point, color=YELLOW, radius=0.08)
                    
                    # Turn previous dot gray
                    gray_dot = Dot(axes.c2p(current_x, value_history[i]), 
                                  color=GRAY, radius=0.06)
                    inner_self.play(Transform(optimization_dot, gray_dot), run_time=0.3)
                    inner_self.add(gray_dot)
                    
                    # Show new position
                    inner_self.play(Create(new_dot), run_time=0.5)
                    optimization_dot = new_dot
                    inner_self.wait(0.5)
                
        config.quality = self.quality
        scene = GradientDescent2DAnimation()
        scene.render()
    

    def _visualize_3d(
        self, 
        param_history: List[np.ndarray], 
        value_history: List[float], 
        gradient_history: List[np.ndarray]
    ) -> None:
        
        """3D visualization for two-parameter optimization"""
        
        
        class GradientDescent3DAnimation(ThreeDScene):
            def construct(inner_self):
                # Define learning rate
                alpha_text = Text(f"α = {self._format_number(self.learning_rate)}", 
                                font_size=24, color=WHITE)
                alpha_text.to_corner(UP + RIGHT)
                inner_self.add_fixed_in_frame_mobjects(alpha_text)
              
                # Get plot function
                plot_function = lambda u,v: self.optimizable.forward(np.array([u,v]))
                
                # Set camera orientation
                inner_self.set_camera_orientation(phi=70 * DEGREES, theta=-90 * DEGREES, 
                                                gamma=0 * DEGREES, distance=10, focal_distance=20)
                
                # Create 3D axes
                axes = ThreeDAxes(
                    x_range=[self.x_range[0], self.x_range[1], self.x_step],
                    y_range=[self.y_range[0], self.y_range[1], self.y_step],
                    z_range=[self.z_range[0], self.z_range[1], self.z_step],
                    x_length=3,
                    y_length=3,
                    z_length=2.5,
                    axis_config={"color": BLUE, "include_numbers": True},
                )
                axes.shift(DOWN * 3.5)
                
                # Create the surface
                def surface_func(u, v):
                    z_val = plot_function(u, v)
                    return axes.c2p(u, v, z_val)
                
                surface = Surface(
                    surface_func,
                    u_range=self.x_range,
                    v_range=self.y_range,
                    resolution=(15, 15),
                    fill_opacity=0.4,
                    stroke_width=0,
                )
                surface.set_fill(BLUE, opacity=0.4)
                                
                # Add axes and surface
                inner_self.play(Create(axes), run_time=1)
                inner_self.play(Create(surface), run_time=1)
                inner_self.wait(0.3)
                
                # Helper function to get surface point
                def get_surface_point(params):
                    x, y = params[0], params[1]
                    z = plot_function(x, y)
                    return axes.c2p(x, y, z)
                
                # Show starting point
                initial_point = get_surface_point(param_history[0])
                optimization_dot = Sphere(radius=0.12, color=YELLOW)
                optimization_dot.set_sheen(0.5, UP)
                optimization_dot.move_to(initial_point)
                inner_self.play(Create(optimization_dot))
                inner_self.wait(0.5)
                
                # Current iteration display (will be updated)
                current_iteration_display = VGroup()
                
                # Begin camera rotation
                inner_self.begin_ambient_camera_rotation(rate=0.1, about="theta")
                
                # Animate through optimization steps
                for i in range(len(param_history) - 1):
                    current_params = param_history[i]
                    gradient_vals = gradient_history[i]
                    next_params = param_history[i + 1]
                    
                    # Clear previous iteration display
                    if len(current_iteration_display) > 0:
                        for obj in current_iteration_display:
                            inner_self.remove(obj)
                    
                    # Create new iteration display - only show current step
                    current_text = Text(f"(x{i}, y{i}) = ({self._format_number(current_params[0])}, {self._format_number(current_params[1])})", 
                                    font_size=24, color=BLUE)
                    current_text.move_to(RIGHT * 4 + UP * 3)
                    
                    grad_x_text = Text(f"df/dx = {self._format_number(gradient_vals[0])}", 
                                    font_size=22, color=GREEN)
                    grad_x_text.next_to(current_text, DOWN, buff=0.2)
                    
                    grad_y_text = Text(f"df/dy = {self._format_number(gradient_vals[1])}", 
                                    font_size=22, color=GREEN)
                    grad_y_text.next_to(grad_x_text, DOWN, buff=0.2)
                    
                    update_x_text = Text(f"x{i+1} = {self._format_number(next_params[0])}", 
                                    font_size=22, color=YELLOW)
                    update_x_text.next_to(grad_y_text, DOWN, buff=0.2)
                    
                    update_y_text = Text(f"y{i+1} = {self._format_number(next_params[1])}", 
                                    font_size=22, color=YELLOW)
                    update_y_text.next_to(update_x_text, DOWN, buff=0.2)
                    
                    current_iteration_display = VGroup(current_text, grad_x_text, grad_y_text, 
                                                    update_x_text, update_y_text)
                    
                    # Add to fixed frame
                    for obj in current_iteration_display:
                        inner_self.add_fixed_in_frame_mobjects(obj)
                    
                    # Show current iteration
                    inner_self.play(Write(current_iteration_display), run_time=0.8)
                    inner_self.wait(1.0)
                    
                    # Update point on surface
                    new_point = get_surface_point(next_params)
                    new_dot = Sphere(radius=0.12, color=YELLOW)
                    new_dot.set_sheen(0.5, UP)
                    new_dot.move_to(new_point)
                    
                    # Turn previous dot gray
                    gray_dot = Sphere(radius=0.08, color=GRAY)
                    gray_dot.set_sheen(0.3, UP)
                    gray_dot.move_to(get_surface_point(current_params))
                    
                    inner_self.play(Transform(optimization_dot, gray_dot), run_time=0.3)
                    inner_self.add(gray_dot)
                    
                    # Show new position
                    inner_self.play(Create(new_dot), run_time=0.5)
                    optimization_dot = new_dot
                    inner_self.wait(0.5)
                
                inner_self.stop_ambient_camera_rotation()
                
                inner_self.wait(3)

        config.quality = self.quality
        config.disable_caching = True # Too many elements in 3d
        
        scene = GradientDescent3DAnimation()
        scene.render()
