import numpy as np
from manim import *



class LinearRegression:
    """
    Linear regression using the closed-form solution.
    Includes support for intercept and animation via Manim.
    """
    
    def __init__(self,fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.weights=None


    def add_bias(self,X):  
        return np.hstack((X, np.ones((X.shape[0], 1))))


    def fit(self,X,y):
        """
        Compute the weights of the closed-form linear regression with input variable X and output variable y
        """
        # Use closed form solution to compute W= (XtX)-1Xt y with X of shape m,p and y of shape m,q, if fit_intercept, add one feature with only one to x (m,p+1)
        # X must be of shape (m,p) and y of shape (m,q)
        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError("Dimensions do not match: X and y must have the same number of samples")
        
        # If fit_intercept, add a column of 1 to X to evaluate the bias when solving
        if self.fit_intercept:
            X_bias = self.add_bias(X)
        else:
            X_bias = X

        # If features linearly dependant, the matrix would be singular and the inverse function will return an error
        inverse=np.linalg.inv(X_bias.T.dot(X_bias))

        self.weights= inverse.dot(X_bias.T).dot(y)

    def predict(self,X):
        """
        Returns the predicted value from the input matrix X
        """

        if self.fit_intercept:
            X_bias = self.add_bias(X)
        else:
            X_bias = X

        if self.weights is None:
            raise ValueError("Model needs to be fit before predicting")
        
        if np.shape(X_bias)[1] != np.shape(self.weights)[0]:
            raise ValueError(f"Dimensions do not match: X must have the same number of features as were used in fitting ({np.shape(self.weights)[0]})")
        
        return(X_bias.dot(self.weights))

    def r2_score(self,X,y):
        """
        Compute r squared between the y predicted and the y provided.
        """

        if self.fit_intercept:
            X_bias = self.add_bias(X)
        else:
            X_bias = X
                           
        if self.weights is None:
            raise ValueError("Model needs to be fit before predicting")

        if np.shape(X_bias)[1] != np.shape(self.weights)[0]:
            raise ValueError(f"Dimensions do not match: X must have the same number of features as were used in fitting ({np.shape(self.weights)[0]})")

        if np.shape(X_bias)[0] != np.shape(y)[0]:
            raise ValueError(f"Dimensions do not match: X and y must have the same number of samples ({np.shape(X)[0]} and {np.shape(y)[0]})")

        if np.shape(y)[1] != np.shape(self.weights)[1]:
            raise ValueError(f"Dimensions do not match: y must represent the same number of output variables as while fitting ({np.shape(self.weights[1])})")
        
        y_pred= self.predict(X)

        r2_score = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

        return r2_score


    def fit_and_animate(self, X, y, quality="low_quality"):
        """
        Compute the weights and create an animation showing the linear regression process
        """
        # First check dimensions like in fit method
        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError("Dimensions do not match: X and y must have the same number of samples")
        
        # Only handle 1D input for visualization
        if X.shape[1] > 1:
            raise ValueError("Animation only supports one feature for visualization")
        
        # Fit the model first
        self.fit(X, y)
        
        # Create animation class
        class LinearRegressionAnimation(Scene):
            def construct(inner_self):
                # Setup using outer class methods
                if self.fit_intercept:
                    X_bias = self.add_bias(X)
                else:
                    X_bias = X

                XT = X_bias.T
                XTX = XT.dot(X_bias)
                XTX_inv = np.linalg.inv(XTX)
                step1 = XTX_inv.dot(XT)
                weights = step1.dot(y)
                w = weights.flatten()

                # Define consistent positions
                LEFT_POS = LEFT * 4
                RIGHT_POS = RIGHT * 2
                UP_POS = UP * 2.5
                DOWN_POS = DOWN * 2
                
                ########################
                # Show X and y matrices with labels
                ########################
                x_matrix = Matrix(np.round(X, 2).tolist()).scale(0.5).shift(LEFT_POS)
                x_label = MathTex("X").next_to(x_matrix, UP)
                y_matrix = Matrix(np.round(y, 2).tolist()).scale(0.5).shift(RIGHT_POS)
                y_label = MathTex("y").next_to(y_matrix, UP)
                
                inner_self.play(Write(x_matrix), Write(x_label), Write(y_matrix), Write(y_label))
                inner_self.wait(0.5)

                if self.fit_intercept:
                    x_aug_matrix = Matrix(np.round(X_bias, 2).tolist()).scale(0.5).move_to(x_matrix.get_center())
                    x_aug_label = MathTex("X_{bias}").next_to(x_aug_matrix, UP)
                    inner_self.play(Transform(x_matrix, x_aug_matrix), Transform(x_label, x_aug_label))
                    inner_self.wait(0.5)

                inner_self.play(FadeOut(x_matrix), FadeOut(x_label), FadeOut(y_matrix), FadeOut(y_label))

                ########################
                # Show formula (stays at top)
                ########################
                eq = MathTex(r"\mathbf{w} = (X^TX)^{-1}X^Ty").to_edge(UP)
                inner_self.play(Write(eq))
                inner_self.wait(1)
                
                ########################
                # Show XT dot X → XTX with labels
                ########################
                XT_matrix = Matrix(np.round(XT, 2).tolist()).scale(0.5).shift(LEFT_POS + UP * 0.5)
                XT_label = MathTex("X^T").next_to(XT_matrix, UP)
                X_matrix = Matrix(np.round(X_bias, 2).tolist()).scale(0.5).shift(RIGHT_POS + UP * 0.5)
                X_label = MathTex("X").next_to(X_matrix, UP)
                
                # Create multiplication sign
                mult_sign = MathTex(r"\times").shift(ORIGIN + UP * 0.5)
                
                inner_self.play(Write(XT_matrix), Write(XT_label), Write(X_matrix), Write(X_label), Write(mult_sign))
                inner_self.wait(0.5)
                
                # Show result below the multiplication
                XTX_matrix = Matrix(np.round(XTX, 2).tolist()).scale(0.5).shift(DOWN * 1.5)
                XTX_label = MathTex("X^TX").next_to(XTX_matrix, UP)
                equals_sign = MathTex("=").next_to(mult_sign, DOWN, buff=0.5)
                
                inner_self.play(Write(equals_sign), Write(XTX_matrix), Write(XTX_label))
                inner_self.wait(0.5)
                
                # Clear the screen but keep formula
                inner_self.play(
                    FadeOut(XT_matrix), FadeOut(XT_label), FadeOut(X_matrix), FadeOut(X_label), 
                    FadeOut(mult_sign), FadeOut(equals_sign), FadeOut(XTX_label),
                    XTX_matrix.animate.move_to(ORIGIN + UP * 0.3)
                )
                
                # Add label back
                XTX_label = MathTex("X^TX").next_to(XTX_matrix, UP)
                inner_self.play(Write(XTX_label))
                inner_self.wait(0.5)

                ########################
                # Show inversion with smooth transition
                ########################
                inversion_arrow = MathTex(r"\downarrow").next_to(XTX_matrix, DOWN, buff=0.5)
                XTX_inv_matrix = Matrix(np.round(XTX_inv, 2).tolist()).scale(0.5).next_to(inversion_arrow, DOWN, buff=0.5)
                inv_label = MathTex(r"(X^TX)^{-1}").next_to(XTX_inv_matrix, UP)

                inner_self.play(Write(inversion_arrow))
                inner_self.play(Write(XTX_inv_matrix), Write(inv_label))
                inner_self.wait(0.5)
                
                # Fade out original matrix and arrow, keep inverse
                inner_self.play(
                    FadeOut(XTX_matrix), FadeOut(XTX_label), FadeOut(inversion_arrow), FadeOut(inv_label),
                    XTX_inv_matrix.animate.move_to(LEFT_POS + UP * 0.5)
                )
                
                # Add label back
                inv_label = MathTex(r"(X^TX)^{-1}").next_to(XTX_inv_matrix, UP)
                inner_self.play(Write(inv_label))
                inner_self.wait(0.5)

                ########################
                # Multiply XTX⁻¹ and XT - with labels
                ########################
                XT_matrix = Matrix(np.round(XT, 2).tolist()).scale(0.5).shift(RIGHT_POS + UP * 0.5)
                XT_label = MathTex("X^T").next_to(XT_matrix, UP)
                mult_sign_2 = MathTex(r"\times").shift(ORIGIN + UP * 0.5)
                
                inner_self.play(Write(XT_matrix), Write(XT_label), Write(mult_sign_2))
                inner_self.wait(0.5)
                
                step1_matrix = Matrix(np.round(step1, 2).tolist()).scale(0.5).shift(DOWN * 1.5)
                step1_label = MathTex(r"(X^TX)^{-1}X^T").next_to(step1_matrix, UP)
                equals_sign_2 = MathTex("=").next_to(mult_sign_2, DOWN, buff=0.5)
                
                inner_self.play(Write(equals_sign_2), Write(step1_matrix), Write(step1_label))
                inner_self.wait(0.5)
                
                # Clear multiplication visualizations
                inner_self.play(
                    FadeOut(XTX_inv_matrix), FadeOut(inv_label), FadeOut(XT_matrix), FadeOut(XT_label),
                    FadeOut(mult_sign_2), FadeOut(equals_sign_2), FadeOut(step1_label),
                    step1_matrix.animate.move_to(LEFT_POS + UP * 0.5)
                )
                
                # Add label back
                step1_label = MathTex(r"(X^TX)^{-1}X^T").next_to(step1_matrix, UP)
                inner_self.play(Write(step1_label))

                ########################
                # Multiply with y → w - final step
                ########################
                y_matrix = Matrix(np.round(y, 2).tolist()).scale(0.5).shift(RIGHT_POS + UP * 0.5)
                y_label = MathTex("y").next_to(y_matrix, UP)
                mult_sign_3 = MathTex(r"\times").shift(ORIGIN + UP * 0.5)
                
                inner_self.play(Write(y_matrix), Write(y_label), Write(mult_sign_3))
                inner_self.wait(0.5)
                
                weights_matrix = Matrix(np.round(weights, 2).tolist()).scale(0.5).shift(DOWN * 1.5)
                weights_label = MathTex(r"\mathbf{w}").next_to(weights_matrix, UP)
                equals_sign_3 = MathTex("=").next_to(mult_sign_3, DOWN, buff=0.5)
                
                inner_self.play(Write(equals_sign_3), Write(weights_matrix), Write(weights_label))
                inner_self.wait(0.5)

                # Clean transition to final weights
                inner_self.play(
                    FadeOut(step1_matrix), FadeOut(step1_label), FadeOut(y_matrix), FadeOut(y_label),
                    FadeOut(mult_sign_3), FadeOut(equals_sign_3), FadeOut(eq), FadeOut(weights_label),
                    weights_matrix.animate.move_to(UP * 2)
                )
                
                # Add label back
                weights_label = MathTex(r"\mathbf{w}").next_to(weights_matrix, UP)
                inner_self.play(Write(weights_label))

                if self.fit_intercept:
                    intercept_label = MathTex(r"\text{Intercept: }" + str(np.round(w[-1], 2)))
                    intercept_label.next_to(weights_matrix, DOWN)
                    inner_self.play(Write(intercept_label))

                inner_self.wait(1)

                ########################
                # Enhanced Visualization with R² Calculation
                ########################
                inner_self.play(FadeOut(weights_matrix), FadeOut(weights_label))
                if self.fit_intercept:
                    inner_self.play(FadeOut(intercept_label))

                # Get data range for better axis scaling - start from 0
                x_min, x_max = 0, max(np.max(X), 10)
                y_min, y_max = 0, max(np.max(y), 10)
                
                # Round up to nice values
                x_max = np.ceil(x_max / 5) * 5
                y_max = np.ceil(y_max / 5) * 5
                
                axes = Axes(
                    x_range=[0, x_max, x_max/10],
                    y_range=[0, y_max, y_max/10],
                    x_length=8,
                    y_length=5,
                    axis_config={
                        "include_numbers": True,
                        "include_tip": True,
                        "numbers_to_exclude": [],
                        "decimal_number_config": {"num_decimal_places": 0}
                    }
                ).to_edge(DOWN)
                
                inner_self.play(Create(axes))

                # Show R² formula at the top
                r2_formula = MathTex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}").to_edge(UP)
                inner_self.play(Write(r2_formula))
                inner_self.wait(1)

                # Create points
                dots = VGroup(*[Dot(axes.c2p(X[i][0], y[i][0]), radius=0.05, color=BLUE) for i in range(len(X))])
                inner_self.play(Create(dots))

                # Create regression line
                x_plot_min = 0
                x_plot_max = x_max
                
                X_plot = np.linspace(x_plot_min, x_plot_max, 100).reshape(-1, 1)
                y_plot = self.predict(X_plot)
                
                reg_line = axes.plot_line_graph(
                    x_values=X_plot.flatten(),
                    y_values=y_plot.flatten(),
                    line_color=YELLOW,
                    add_vertex_dots=False
                )
                inner_self.play(Create(reg_line))

                ########################
                # Step 1: Show residuals only
                ########################
                y_pred = self.predict(X)
                residuals = VGroup(*[
                    Line(
                        axes.c2p(X[i][0], y[i][0]), 
                        axes.c2p(X[i][0], y_pred[i][0]), 
                        color=RED,
                        stroke_width=2
                    )
                    for i in range(len(X))
                ])
                
                residuals_label = Text("Residuals", color=RED).scale(0.5).next_to(r2_formula, DOWN, buff=0.3)
                inner_self.play(Create(residuals), Write(residuals_label))
                inner_self.wait(1.5)
                
                # Step 2: Square residuals (show as squared length lines)
                squared_residuals = VGroup()
                for i in range(len(X)):
                    residual_value = y[i][0] - y_pred[i][0]
                    squared_value = residual_value ** 2
                    
                    # Determine direction of squared line (always start from predicted point)
                    if residual_value >= 0:  # Point is above prediction
                        start_point = axes.c2p(X[i][0], y_pred[i][0])
                        end_point = axes.c2p(X[i][0], y_pred[i][0] + squared_value)
                    else:  # Point is below prediction
                        start_point = axes.c2p(X[i][0], y_pred[i][0])
                        end_point = axes.c2p(X[i][0], y_pred[i][0] - squared_value)
                    
                    # Create line for squared residual
                    squared_line = Line(
                        start_point,
                        end_point,
                        color=RED,
                        stroke_width=3  # Thicker to differentiate from regular residuals
                    )
                    
                    squared_residuals.add(squared_line)
                
                ss_res = np.sum((y - y_pred) ** 2)
                squared_label = Text(f"Residuals squared = {np.round(ss_res, 2)}", color=RED).scale(0.5).next_to(r2_formula, DOWN, buff=0.3)
                
                inner_self.play(
                    Transform(residuals, squared_residuals),
                    Transform(residuals_label, squared_label)
                )
                inner_self.wait(1.5)
                
                # Step 3: Update formula with SS_res value
                r2_formula_with_ssres = MathTex(
                    f"R^2 = 1 - \\frac{{{np.round(ss_res, 2)}}}{{SS_{{tot}}}}"
                ).to_edge(UP)
                inner_self.play(Transform(r2_formula, r2_formula_with_ssres))
                inner_self.wait(1)
                
                # Step 4: Hide squared residuals
                inner_self.play(FadeOut(residuals), FadeOut(residuals_label))
                inner_self.wait(0.5)
                
                ########################
                # Step 5: Show total deviations only
                ########################
                y_mean = np.mean(y)
                mean_line = axes.plot(lambda x: y_mean, x_range=[0, x_max], color=GREEN)
                mean_label = MathTex(r"\bar{y}", color=GREEN).next_to(axes.c2p(x_max, y_mean), RIGHT)
                
                inner_self.play(Create(mean_line), Write(mean_label))
                inner_self.wait(0.5)
                
                # Show deviations from mean
                total_deviations = VGroup(*[
                    Line(
                        axes.c2p(X[i][0], y[i][0]), 
                        axes.c2p(X[i][0], y_mean), 
                        color=GREEN,
                        stroke_width=2
                    )
                    for i in range(len(X))
                ])
                
                totals_label = Text("Total Deviations", color=GREEN).scale(0.5).next_to(r2_formula, DOWN, buff=0.3)
                inner_self.play(Create(total_deviations), Write(totals_label))
                inner_self.wait(1.5)
                
                # Step 6: Square total deviations (show as squared length lines)
                squared_total_deviations = VGroup()
                for i in range(len(X)):
                    deviation_value = y[i][0] - y_mean
                    squared_value = deviation_value ** 2
                    
                    # Determine direction of squared line (always start from mean)
                    if deviation_value >= 0:  # Point is above mean
                        start_point = axes.c2p(X[i][0], y_mean)
                        end_point = axes.c2p(X[i][0], y_mean + squared_value)
                    else:  # Point is below mean
                        start_point = axes.c2p(X[i][0], y_mean)
                        end_point = axes.c2p(X[i][0], y_mean - squared_value)
                    
                    # Create line for squared deviation
                    squared_line = Line(
                        start_point,
                        end_point,
                        color=GREEN,
                        stroke_width=3  # Thicker to differentiate from regular deviations
                    )
                    
                    squared_total_deviations.add(squared_line)
                
                ss_tot = np.sum((y - y_mean) ** 2)
                totals_squared_label = Text(f"Total Deviations squared = {np.round(ss_tot, 2)}", color=GREEN).scale(0.5).next_to(r2_formula, DOWN, buff=0.3)
                
                inner_self.play(
                    Transform(total_deviations, squared_total_deviations),
                    Transform(totals_label, totals_squared_label)
                )
                inner_self.wait(1.5)
                
                # Step 7: Update formula with SS_tot value
                r2_formula_complete = MathTex(
                    f"R^2 = 1 - \\frac{{{np.round(ss_res, 2)}}}{{{np.round(ss_tot, 2)}}}"
                ).to_edge(UP)
                inner_self.play(Transform(r2_formula, r2_formula_complete))
                inner_self.wait(1)
                
                ########################
                # Step 8: Calculate final R²
                ########################
                r2 = self.r2_score(X, y)
                r2_final = MathTex(f"R^2 = {np.round(r2, 3)}").scale(1.5).move_to(ORIGIN)
                
                inner_self.play(
                    FadeOut(total_deviations),
                    FadeOut(totals_label),
                    FadeOut(mean_line),
                    FadeOut(mean_label),
                    FadeOut(r2_formula),
                    Write(r2_final)
                )
                inner_self.wait(2)

        # Fix quality mapping
        from manim import config
        config.quality = quality

        # Render scene
        scene = LinearRegressionAnimation()
        scene.render()