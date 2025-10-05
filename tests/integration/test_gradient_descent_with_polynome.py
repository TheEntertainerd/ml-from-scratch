import numpy as np
from src.utils.polynomial import MultivariatePolynomialOptimizable, MultivariatePolynomialVisualizer
from src.gradient_descent import GradientDescent
import pytest
from scipy.optimize import minimize
from pathlib import Path


def test_gradient_descent_matches_scipy_simple() -> None:
    poly_dict: dict[tuple[int, ...], float] = {
        (2, 0): 1,  # x²
        (0, 2): 2,  # 2y²
    }
    start = np.array([-1.2, 1.0])
    poly = MultivariatePolynomialOptimizable(poly_dict, start.copy())
    gd = GradientDescent(
        learning_rate=0.1,
        max_iterations=500,
        optimizable=poly,
        tolerance=1e-6,
    )
    gd_params, gd_value = gd.find_optimal()

    def forward_wrapper(params: np.ndarray) -> float:
        poly.set_params(params)
        return poly.forward()

    def gradient_wrapper(params: np.ndarray) -> np.ndarray:
        poly.set_params(params)
        return poly.gradient()

    res = minimize(forward_wrapper, start, jac=gradient_wrapper, tol=1e-6)
    np.testing.assert_allclose(gd_params, res.x, atol=1e-5)
    np.testing.assert_allclose(gd_value, res.fun, atol=1e-5)


def test_gradient_descent_matches_scipy_complex() -> None:
    poly_dict: dict[tuple[int, ...], float] = {
        (2, 0, 0): 1,  # x²
        (0, 2, 0): 2,  # 2y²
        (1, 1, 1): 4,  # 4xyz
        (0, 0, 2): 3,  # 3z²
        (0, 0, 0): 5,  # 5
    }
    start = np.array([-1.2, 1.0, 0.5])
    poly = MultivariatePolynomialOptimizable(poly_dict, start.copy())
    gd = GradientDescent(
        learning_rate=0.1,
        max_iterations=500,
        optimizable=poly,
        tolerance=1e-6,
    )
    gd_params, gd_value = gd.find_optimal()

    def forward_wrapper(params: np.ndarray) -> float:
        poly.set_params(params)
        return poly.forward()

    def gradient_wrapper(params: np.ndarray) -> np.ndarray:
        poly.set_params(params)
        return poly.gradient()

    res = minimize(forward_wrapper, start, jac=gradient_wrapper, tol=1e-6)
    np.testing.assert_allclose(gd_params, res.x, atol=1e-5)
    np.testing.assert_allclose(gd_value, res.fun, atol=1e-5)


@pytest.mark.slow
def test_find_optimal_with_visualizer_create_video_2_variables(tmp_path: Path) -> None:
    output_dir = tmp_path / "video_out"
    polynome_dictionary_2d: dict[tuple[int, ...], float] = {
        (2, 0): 2,  # 2x²
        (0, 2): 1,  # y²
    }
    initial_parameters_2d = np.array([2.0, 2.0])
    polynome_2d = MultivariatePolynomialOptimizable(polynome_dictionary_2d, initial_parameters_2d)
    visualizer = MultivariatePolynomialVisualizer(
        optimizable=polynome_2d,
        learning_rate=0.1,
        x_range=(-3, 3),
        y_range=(-3, 3),
        z_range=(-20, 20),
        x_step=1,
        y_step=1,
        z_step=5,
        quality="low_quality",
        output_dir=str(output_dir),
    )
    gd_with_vis = GradientDescent(
        learning_rate=0.1,
        max_iterations=3,
        optimizable=polynome_2d,
        tolerance=1e-6,
        visualizer=visualizer,
    )
    _ = gd_with_vis.find_optimal()
    assert True


@pytest.mark.slow
def test_find_optimal_with_visualizer_create_video_1_variable(tmp_path: Path) -> None:
    output_dir = tmp_path / "video_out"
    polynome_dic_1d: dict[tuple[int, ...], float] = {
        (4,): 1,  # x⁴
        (3,): -4,  # -4x³
        (2,): -3,  # -3x²
        (1,): 2,  # 2x
        (0,): 1,  # constant term
    }
    initial_parameters_1d = np.array([2])
    optimizable = MultivariatePolynomialOptimizable(polynome_dic_1d, initial_parameters_1d)
    visualizer = MultivariatePolynomialVisualizer(
        optimizable,
        learning_rate=0.01,
        x_range=(-5, 5),
        y_range=(-50, 50),
        z_range=(-100, 100),
        x_step=1,
        y_step=50,
        z_step=50,
        quality="low_quality",
        output_dir=str(output_dir),
    )
    gd_with_vis = GradientDescent(
        learning_rate=0.01,
        max_iterations=2,
        optimizable=optimizable,
        tolerance=1e-6,
        visualizer=visualizer,
    )
    _ = gd_with_vis.find_optimal()
    assert True


def test_find_optimal_with_visualizer_raises_on_more_than_one_feature(tmp_path: Path) -> None:
    output_dir = tmp_path / "video_out"
    polynome_dictionary_2d: dict[tuple[int, ...], float] = {
        (2, 0, 0): 2,  # 2x²
        (0, 2, 0): 1,  # y²
        (0, 0, 1): 1,  # z
    }
    initial_parameters_2d = np.array([2.0, 2.0, 1.0])
    polynome_2d = MultivariatePolynomialOptimizable(polynome_dictionary_2d, initial_parameters_2d)
    visualizer = MultivariatePolynomialVisualizer(
        optimizable=polynome_2d,
        learning_rate=0.1,
        x_range=(-3, 3),
        y_range=(-3, 3),
        z_range=(-20, 20),
        x_step=1,
        y_step=1,
        z_step=5,
        quality="low_quality",
        output_dir=str(output_dir),
    )
    gd_with_vis = GradientDescent(
        learning_rate=0.1,
        max_iterations=3,
        optimizable=polynome_2d,
        tolerance=1e-6,
        visualizer=visualizer,
    )
    with pytest.raises(ValueError, match="Cannot visualize optimization for polynome of"):
        _ = gd_with_vis.find_optimal()
