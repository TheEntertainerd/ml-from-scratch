import numpy as np
from src.gradient_descent import GradientDescent
from src.utils.polynomial import Optimizable
from unittest.mock import MagicMock
from _pytest.capture import CaptureFixture


def test_gradient_descent_runs_only_once_on_zero_gradient() -> None:
    class SimpleOpt(Optimizable):
        def __init__(self, params: np.ndarray):
            super().__init__(params)
            self.calls = 0

        def gradient(self) -> np.ndarray:
            self.calls += 1  # Count number of calls
            return np.zeros_like(self.parameters)

        def forward(self) -> float:
            return 0

    start_params = np.array([1.0])
    opt = SimpleOpt(start_params)
    gd = GradientDescent(learning_rate=0.1, max_iterations=50, optimizable=opt)
    final_params, _ = gd.find_optimal()
    assert opt.calls == 1
    np.testing.assert_allclose(final_params, start_params)


def test_gradient_descent_stops_on_tolerance() -> None:
    class SimpleOpt(Optimizable):
        def gradient(self) -> np.ndarray:
            return self.parameters  # grad = x

        def forward(self) -> float:
            return float(0.5 * np.sum(self.parameters**2))

    start_params = np.array([1e-8])
    optimizable = SimpleOpt(start_params.copy())
    gd = GradientDescent(learning_rate=0.1, max_iterations=50, optimizable=optimizable, tolerance=1e-6)
    final_params, final_value = gd.find_optimal()
    # Stops after first update
    np.testing.assert_allclose(final_params, start_params * (1 - gd.learning_rate))
    np.testing.assert_allclose(final_value, 0.5 * np.sum(final_params**2))


def test_gradient_descent_verbosity_prints(capsys: CaptureFixture[str]) -> None:
    class SimpleOpt(Optimizable):
        def gradient(self) -> np.ndarray:
            return np.array([1.0])

        def forward(self) -> float:
            return float(self.parameters[0])

    opt = SimpleOpt(np.array([1.0]))
    gd = GradientDescent(learning_rate=0.1, max_iterations=1, optimizable=opt, verbosity=1)
    gd.find_optimal()
    captured = capsys.readouterr()
    assert "Gradient =" in captured.out


def test_gradient_descent_calls_visualizer() -> None:
    class SimpleOpt(Optimizable):
        def gradient(self) -> np.ndarray:
            return np.array([1.0])

        def forward(self) -> float:
            return float(self.parameters[0])

    opt = SimpleOpt(np.array([1.0]))
    mock_vis = MagicMock()
    gd = GradientDescent(learning_rate=0.1, max_iterations=2, optimizable=opt, visualizer=mock_vis)
    gd.find_optimal()
    mock_vis.visualize.assert_called_once()
