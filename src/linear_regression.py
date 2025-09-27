import numpy as np
import manim as mn


class LinearRegression:
    """
    Linear regression using the closed-form solution.
    Includes support for intercept and animation via Manim.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.weights = None

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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
        inverse = np.linalg.inv(X_bias.T.dot(X_bias))

        self.weights = inverse.dot(X_bias.T).dot(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
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
            raise ValueError(
                f"Dimensions do not match: X must have the same number of features as were used in fitting ({np.shape(self.weights)[0]})"
            )

        return X_bias.dot(self.weights)

    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
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
            raise ValueError(
                f"Dimensions do not match: X must have the same number of features as were used in fitting ({np.shape(self.weights)[0]})"
            )

        if np.shape(X_bias)[0] != np.shape(y)[0]:
            raise ValueError(
                f"Dimensions do not match: X and y must have the same number of samples ({np.shape(X)[0]} and {np.shape(y)[0]})"
            )

        if np.shape(y)[1] != np.shape(self.weights)[1]:
            raise ValueError(
                f"Dimensions do not match: y must represent the same number of output variables as while fitting ({np.shape(self.weights[1])})"
            )

        y_pred = self.predict(X)

        r2_score = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

        return r2_score

    def fit_and_animate(self, X: np.ndarray, y: np.ndarray, quality: str = "low_quality", output_dir: str = "") -> None:
        """
        Compute the weights and create an animation showing the linear regression process


        Args:
            X: Input features
            y: Target values
            quality: Animation quality ("low_quality", "medium_quality", "high_quality", etc.)
            output_dir: Directory to save the animation output. If None, uses default manim directory.
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
        class LinearRegressionAnimation(mn.Scene):
            def construct(inner_self) -> None:
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
                LEFT_POS = mn.LEFT * 4
                RIGHT_POS = mn.RIGHT * 2

                ########################
                # Show X and y matrices with labels
                ########################
                x_matrix = mn.Matrix(np.round(X, 2).tolist()).scale(0.5).shift(LEFT_POS)
                x_label = mn.MathTex("X").next_to(x_matrix, mn.UP)
                y_matrix = mn.Matrix(np.round(y, 2).tolist()).scale(0.5).shift(RIGHT_POS)
                y_label = mn.MathTex("y").next_to(y_matrix, mn.UP)

                inner_self.play(mn.Write(x_matrix), mn.Write(x_label), mn.Write(y_matrix), mn.Write(y_label))
                inner_self.wait(0.5)

                if self.fit_intercept:
                    x_aug_matrix = mn.Matrix(np.round(X_bias, 2).tolist()).scale(0.5).move_to(x_matrix.get_center())
                    x_aug_label = mn.MathTex("X_{bias}").next_to(x_aug_matrix, mn.UP)
                    inner_self.play(mn.Transform(x_matrix, x_aug_matrix), mn.Transform(x_label, x_aug_label))
                    inner_self.wait(0.5)

                inner_self.play(mn.FadeOut(x_matrix), mn.FadeOut(x_label), mn.FadeOut(y_matrix), mn.FadeOut(y_label))

                ########################
                # Show formula (stays at top)
                ########################
                eq = mn.MathTex(r"\mathbf{w} = (X^TX)^{-1}X^Ty").to_edge(mn.UP)
                inner_self.play(mn.Write(eq))
                inner_self.wait(1)

                ########################
                # Show XT dot X → XTX with labels
                ########################
                XT_matrix = mn.Matrix(np.round(XT, 2).tolist()).scale(0.5).shift(LEFT_POS + mn.UP * 0.5)
                XT_label = mn.MathTex("X^T").next_to(XT_matrix, mn.UP)
                X_matrix = mn.Matrix(np.round(X_bias, 2).tolist()).scale(0.5).shift(RIGHT_POS + mn.UP * 0.5)
                X_label = mn.MathTex("X").next_to(X_matrix, mn.UP)

                # Create multiplication sign
                mult_sign = mn.MathTex(r"\times").shift(mn.ORIGIN + mn.UP * 0.5)

                inner_self.play(
                    mn.Write(XT_matrix), mn.Write(XT_label), mn.Write(X_matrix), mn.Write(X_label), mn.Write(mult_sign)
                )
                inner_self.wait(0.5)

                # Show result below the multiplication
                XTX_matrix = mn.Matrix(np.round(XTX, 2).tolist()).scale(0.5).shift(mn.DOWN * 1.5)
                XTX_label = mn.MathTex("X^TX").next_to(XTX_matrix, mn.UP)
                equals_sign = mn.MathTex("=").next_to(mult_sign, mn.DOWN, buff=0.5)

                inner_self.play(mn.Write(equals_sign), mn.Write(XTX_matrix), mn.Write(XTX_label))
                inner_self.wait(0.5)

                # Clear the screen but keep formula
                inner_self.play(
                    mn.FadeOut(XT_matrix),
                    mn.FadeOut(XT_label),
                    mn.FadeOut(X_matrix),
                    mn.FadeOut(X_label),
                    mn.FadeOut(mult_sign),
                    mn.FadeOut(equals_sign),
                    mn.FadeOut(XTX_label),
                    XTX_matrix.animate.move_to(mn.ORIGIN + mn.UP * 0.3),
                )

                # Add label back
                XTX_label = mn.MathTex("X^TX").next_to(XTX_matrix, mn.UP)
                inner_self.play(mn.Write(XTX_label))
                inner_self.wait(0.5)

                ########################
                # Show inversion with smooth transition
                ########################
                inversion_arrow = mn.MathTex(r"\downarrow").next_to(XTX_matrix, mn.DOWN, buff=0.5)
                XTX_inv_matrix = (
                    mn.Matrix(np.round(XTX_inv, 2).tolist()).scale(0.5).next_to(inversion_arrow, mn.DOWN, buff=0.5)
                )
                inv_label = mn.MathTex(r"(X^TX)^{-1}").next_to(XTX_inv_matrix, mn.UP)

                inner_self.play(mn.Write(inversion_arrow))
                inner_self.play(mn.Write(XTX_inv_matrix), mn.Write(inv_label))
                inner_self.wait(0.5)

                # Fade out original matrix and arrow, keep inverse
                inner_self.play(
                    mn.FadeOut(XTX_matrix),
                    mn.FadeOut(XTX_label),
                    mn.FadeOut(inversion_arrow),
                    mn.FadeOut(inv_label),
                    XTX_inv_matrix.animate.move_to(LEFT_POS + mn.UP * 0.5),
                )

                # Add label back
                inv_label = mn.MathTex(r"(X^TX)^{-1}").next_to(XTX_inv_matrix, mn.UP)
                inner_self.play(mn.Write(inv_label))
                inner_self.wait(0.5)

                ########################
                # Multiply XTX⁻¹ and XT - with labels
                ########################
                XT_matrix = mn.Matrix(np.round(XT, 2).tolist()).scale(0.5).shift(RIGHT_POS + mn.UP * 0.5)
                XT_label = mn.MathTex("X^T").next_to(XT_matrix, mn.UP)
                mult_sign_2 = mn.MathTex(r"\times").shift(mn.ORIGIN + mn.UP * 0.5)

                inner_self.play(mn.Write(XT_matrix), mn.Write(XT_label), mn.Write(mult_sign_2))
                inner_self.wait(0.5)

                step1_matrix = mn.Matrix(np.round(step1, 2).tolist()).scale(0.5).shift(mn.DOWN * 1.5)
                step1_label = mn.MathTex(r"(X^TX)^{-1}X^T").next_to(step1_matrix, mn.UP)
                equals_sign_2 = mn.MathTex("=").next_to(mult_sign_2, mn.DOWN, buff=0.5)

                inner_self.play(mn.Write(equals_sign_2), mn.Write(step1_matrix), mn.Write(step1_label))
                inner_self.wait(0.5)

                # Clear multiplication visualizations
                inner_self.play(
                    mn.FadeOut(XTX_inv_matrix),
                    mn.FadeOut(inv_label),
                    mn.FadeOut(XT_matrix),
                    mn.FadeOut(XT_label),
                    mn.FadeOut(mult_sign_2),
                    mn.FadeOut(equals_sign_2),
                    mn.FadeOut(step1_label),
                    step1_matrix.animate.move_to(LEFT_POS + mn.UP * 0.5),
                )

                # Add label back
                step1_label = mn.MathTex(r"(X^TX)^{-1}X^T").next_to(step1_matrix, mn.UP)
                inner_self.play(mn.Write(step1_label))

                ########################
                # Multiply with y → w - final step
                ########################
                y_matrix = mn.Matrix(np.round(y, 2).tolist()).scale(0.5).shift(RIGHT_POS + mn.UP * 0.5)
                y_label = mn.MathTex("y").next_to(y_matrix, mn.UP)
                mult_sign_3 = mn.MathTex(r"\times").shift(mn.ORIGIN + mn.UP * 0.5)

                inner_self.play(mn.Write(y_matrix), mn.Write(y_label), mn.Write(mult_sign_3))
                inner_self.wait(0.5)

                weights_matrix = mn.Matrix(np.round(weights, 2).tolist()).scale(0.5).shift(mn.DOWN * 1.5)
                weights_label = mn.MathTex(r"\mathbf{w}").next_to(weights_matrix, mn.UP)
                equals_sign_3 = mn.MathTex("=").next_to(mult_sign_3, mn.DOWN, buff=0.5)

                inner_self.play(mn.Write(equals_sign_3), mn.Write(weights_matrix), mn.Write(weights_label))
                inner_self.wait(0.5)

                # Clean transition to final weights
                inner_self.play(
                    mn.FadeOut(step1_matrix),
                    mn.FadeOut(step1_label),
                    mn.FadeOut(y_matrix),
                    mn.FadeOut(y_label),
                    mn.FadeOut(mult_sign_3),
                    mn.FadeOut(equals_sign_3),
                    mn.FadeOut(eq),
                    mn.FadeOut(weights_label),
                    weights_matrix.animate.move_to(mn.UP * 2),
                )

                # Add label back
                weights_label = mn.MathTex(r"\mathbf{w}").next_to(weights_matrix, mn.UP)
                inner_self.play(mn.Write(weights_label))

                if self.fit_intercept:
                    intercept_label = mn.MathTex(r"\text{Intercept: }" + str(np.round(w[-1], 2)))
                    intercept_label.next_to(weights_matrix, mn.DOWN)
                    inner_self.play(mn.Write(intercept_label))

                inner_self.wait(1)

                ########################
                # Enhanced Visualization with R² Calculation
                ########################
                inner_self.play(mn.FadeOut(weights_matrix), mn.FadeOut(weights_label))
                if self.fit_intercept:
                    inner_self.play(mn.FadeOut(intercept_label))

                # Get data range for better axis scaling - start from 0
                x_min, x_max = np.min(X), np.max(X)
                y_min, y_max = np.min(y), np.max(y)

                # Add padding (10% on each side)
                x_padding = (x_max - x_min) * 0.1
                y_padding = (y_max - y_min) * 0.1
                x_min -= x_padding
                x_max += x_padding
                y_min -= y_padding
                y_max += y_padding

                # Round to nice values
                x_range = x_max - x_min
                y_range = y_max - y_min
                x_step = np.ceil(x_range / 10)
                y_step = np.ceil(y_range / 10)
                x_min = np.floor(x_min / x_step) * x_step
                x_max = np.ceil(x_max / x_step) * x_step
                y_min = np.floor(y_min / y_step) * y_step
                y_max = np.ceil(y_max / y_step) * y_step

                axes = mn.Axes(
                    x_range=[x_min, x_max, x_step],
                    y_range=[y_min, y_max, y_step],
                    x_length=8,
                    y_length=5,
                    axis_config={
                        "include_numbers": True,
                        "include_tip": True,
                        "numbers_to_exclude": [],
                        "decimal_number_config": {"num_decimal_places": 1},
                    },
                ).to_edge(mn.DOWN)

                inner_self.play(mn.Create(axes))

                # Show R² formula at the top
                r2_formula = mn.MathTex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}").to_edge(mn.UP)
                inner_self.play(mn.Write(r2_formula))
                inner_self.wait(1)

                # Create points
                dots = mn.VGroup(
                    *[mn.Dot(axes.c2p(X[i][0], y[i][0]), radius=0.05, color=mn.BLUE) for i in range(len(X))]
                )
                inner_self.play(mn.Create(dots))

                # Create regression line
                X_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1)
                y_plot = self.predict(X_plot)

                reg_line = axes.plot_line_graph(
                    x_values=X_plot.flatten(), y_values=y_plot.flatten(), line_color=mn.YELLOW, add_vertex_dots=False
                )
                inner_self.play(mn.Create(reg_line))

                ########################
                # Step 1: Show residuals only
                ########################
                y_pred = self.predict(X)
                residuals = mn.VGroup(
                    *[
                        mn.Line(
                            axes.c2p(X[i][0], y[i][0]), axes.c2p(X[i][0], y_pred[i][0]), color=mn.RED, stroke_width=2
                        )
                        for i in range(len(X))
                    ]
                )

                residuals_label = mn.Text("Residuals", color=mn.RED).scale(0.5).next_to(r2_formula, mn.DOWN, buff=0.3)
                inner_self.play(mn.Create(residuals), mn.Write(residuals_label))
                inner_self.wait(1.5)

                # Step 2: Square residuals (show as squared length lines)
                squared_residuals = mn.VGroup()
                for i in range(len(X)):
                    residual_value = y[i][0] - y_pred[i][0]
                    squared_value = residual_value**2

                    # Determine direction of squared line (always start from predicted point)
                    if residual_value >= 0:  # Point is above prediction
                        start_point = axes.c2p(X[i][0], y_pred[i][0])
                        end_point = axes.c2p(X[i][0], y_pred[i][0] + squared_value)
                    else:  # Point is below prediction
                        start_point = axes.c2p(X[i][0], y_pred[i][0])
                        end_point = axes.c2p(X[i][0], y_pred[i][0] - squared_value)

                    # Create line for squared residual
                    squared_line = mn.Line(
                        start_point,
                        end_point,
                        color=mn.RED,
                        stroke_width=3,  # Thicker to differentiate from regular residuals
                    )

                    squared_residuals.add(squared_line)

                ss_res = np.sum((y - y_pred) ** 2)
                squared_label = (
                    mn.Text(f"Residuals squared = {np.round(ss_res, 2)}", color=mn.RED)
                    .scale(0.5)
                    .next_to(r2_formula, mn.DOWN, buff=0.3)
                )

                inner_self.play(
                    mn.Transform(residuals, squared_residuals), mn.Transform(residuals_label, squared_label)
                )
                inner_self.wait(1.5)

                # Step 3: Update formula with SS_res value
                r2_formula_with_ssres = mn.MathTex(f"R^2 = 1 - \\frac{{{np.round(ss_res, 2)}}}{{SS_{{tot}}}}").to_edge(
                    mn.UP
                )
                inner_self.play(mn.Transform(r2_formula, r2_formula_with_ssres))
                inner_self.wait(1)

                # Step 4: Hide squared residuals
                inner_self.play(mn.FadeOut(residuals), mn.FadeOut(residuals_label))
                inner_self.wait(0.5)

                ########################
                # Step 5: Show total deviations only
                ########################
                y_mean = np.mean(y)
                mean_line = axes.plot(lambda x: y_mean, x_range=[x_min, x_max], color=mn.GREEN)
                mean_label = mn.MathTex(r"\bar{y}", color=mn.GREEN).next_to(axes.c2p(x_max, y_mean), mn.RIGHT)

                inner_self.play(mn.Create(mean_line), mn.Write(mean_label))
                inner_self.wait(0.5)

                # Show deviations from mean
                total_deviations = mn.VGroup(
                    *[
                        mn.Line(axes.c2p(X[i][0], y[i][0]), axes.c2p(X[i][0], y_mean), color=mn.GREEN, stroke_width=2)
                        for i in range(len(X))
                    ]
                )

                totals_label = (
                    mn.Text("Total Deviations", color=mn.GREEN).scale(0.5).next_to(r2_formula, mn.DOWN, buff=0.3)
                )
                inner_self.play(mn.Create(total_deviations), mn.Write(totals_label))
                inner_self.wait(1.5)

                # Step 6: Square total deviations (show as squared length lines)
                squared_total_deviations = mn.VGroup()
                for i in range(len(X)):
                    deviation_value = y[i][0] - y_mean
                    squared_value = deviation_value**2

                    # Determine direction of squared line (always start from mean)
                    if deviation_value >= 0:  # Point is above mean
                        start_point = axes.c2p(X[i][0], y_mean)
                        end_point = axes.c2p(X[i][0], y_mean + squared_value)
                    else:  # Point is below mean
                        start_point = axes.c2p(X[i][0], y_mean)
                        end_point = axes.c2p(X[i][0], y_mean - squared_value)

                    # Create line for squared deviation
                    squared_line = mn.Line(
                        start_point,
                        end_point,
                        color=mn.GREEN,
                        stroke_width=3,  # Thicker to differentiate from regular deviations
                    )

                    squared_total_deviations.add(squared_line)

                ss_tot = np.sum((y - y_mean) ** 2)
                totals_squared_label = (
                    mn.Text(f"Total Deviations squared = {np.round(ss_tot, 2)}", color=mn.GREEN)
                    .scale(0.5)
                    .next_to(r2_formula, mn.DOWN, buff=0.3)
                )

                inner_self.play(
                    mn.Transform(total_deviations, squared_total_deviations),
                    mn.Transform(totals_label, totals_squared_label),
                )
                inner_self.wait(1.5)

                # Step 7: Update formula with SS_tot value
                r2_formula_complete = mn.MathTex(
                    f"R^2 = 1 - \\frac{{{np.round(ss_res, 2)}}}{{{np.round(ss_tot, 2)}}}"
                ).to_edge(mn.UP)
                inner_self.play(mn.Transform(r2_formula, r2_formula_complete))
                inner_self.wait(1)

                ########################
                # Step 8: Calculate final R²
                ########################
                r2 = self.r2_score(X, y)
                r2_final = mn.MathTex(f"R^2 = {np.round(r2, 3)}").scale(1.5).move_to(mn.ORIGIN)

                inner_self.play(
                    mn.FadeOut(total_deviations),
                    mn.FadeOut(totals_label),
                    mn.FadeOut(mean_line),
                    mn.FadeOut(mean_label),
                    mn.FadeOut(r2_formula),
                    mn.Write(r2_final),
                )
                inner_self.wait(2)

        # Fix quality mapping
        from manim import config

        config.quality = quality

        # Set output directory if specified
        if output_dir:
            import os

            os.makedirs(output_dir, exist_ok=True)
            config.media_dir = str(output_dir)

        # Render scene
        scene = LinearRegressionAnimation()
        scene.render()
