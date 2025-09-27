import numpy as np
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn
from src.linear_regression import LinearRegression as LinearRegressionNumpy
import pytest
from unittest.mock import patch
from pathlib import Path
from tests.types import DatasetTrainTest


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_r2_score(synthetic_data: DatasetTrainTest, fit_intercept: bool) -> None:
    X_train, y_train, X_test, y_test = synthetic_data
    lr_sklearn = LinearRegressionSklearn(fit_intercept=fit_intercept)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    lr_sklearn.fit(X_train, y_train)
    score_sklearn = lr_sklearn.score(X_test, y_test)
    score_numpy = lr_numpy.r2_score(X_test, y_test)
    np.testing.assert_allclose(score_numpy, score_sklearn)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_r2_score_raises_on_model_not_fit(synthetic_data: DatasetTrainTest, fit_intercept: bool) -> None:
    _, _, X_test, y_test = synthetic_data
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="Model needs to be fit before predicting"):
        lr_numpy.r2_score(X_test, y_test)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_r2_score_raises_on_mismatch_dimensions_Xtest_Xtrain(fit_intercept: bool) -> None:
    X_train = np.random.randn(7, 5)
    y_train = 2 * X_train + 1
    X_test = np.random.randn(3, 3)
    y_test = np.random.randn(3, 1)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    with pytest.raises(
        ValueError, match="Dimensions do not match: X must have the same number of features as were used in fitting"
    ):
        lr_numpy.r2_score(X_test, y_test)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_r2_score_raises_on_mismatch_dimensions_X_y(fit_intercept: bool) -> None:
    X_train = np.random.randn(8, 3)
    y_train = np.random.randn(8, 2)
    X_test = np.random.randn(8, 3)
    y_test = np.random.randn(7, 3)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    with pytest.raises(ValueError, match="Dimensions do not match: X and y must have the same number of samples"):
        lr_numpy.r2_score(X_test, y_test)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_r2_score_raises_on_mismatch_dimensions_y(fit_intercept: bool) -> None:
    X_train = np.random.randn(8, 3)
    y_train = np.random.randn(8, 2)
    X_test = np.random.randn(8, 3)
    y_test = np.random.randn(8, 3)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    with pytest.raises(
        ValueError,
        match="Dimensions do not match: y must represent the same number of output variables as while fitting",
    ):
        lr_numpy.r2_score(X_test, y_test)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_predict(synthetic_data: DatasetTrainTest, fit_intercept: bool) -> None:
    X_train, y_train, X_test, _ = synthetic_data
    lr_sklearn = LinearRegressionSklearn(fit_intercept=fit_intercept)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    lr_sklearn.fit(X_train, y_train)
    prediction_numpy = lr_numpy.predict(X_test)
    prediction_sklearn = lr_sklearn.predict(X_test)
    np.testing.assert_allclose(prediction_numpy, prediction_sklearn)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_predict_raises_on_model_not_fit(synthetic_data: DatasetTrainTest, fit_intercept: bool) -> None:
    _, _, X_test, _ = synthetic_data
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="Model needs to be fit before predicting"):
        lr_numpy.predict(X_test)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_predict_raises_on_mismatched_shapes(fit_intercept: bool) -> None:
    np.random.seed(42)
    X_train = np.random.randn(7, 5)
    y_train = 2 * X_train + 1
    X_test = np.random.randn(3, 3)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    with pytest.raises(ValueError, match="Dimensions do not match"):
        lr_numpy.predict(X_test)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit(synthetic_data: DatasetTrainTest, fit_intercept: bool) -> None:
    X_train, y_train, _, _ = synthetic_data
    lr_sklearn = LinearRegressionSklearn(fit_intercept=fit_intercept)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    lr_sklearn.fit(X_train, y_train)
    assert lr_numpy.weights is not None
    weights_numpy = np.squeeze(lr_numpy.weights)
    if fit_intercept:
        weights_sklearn = np.concatenate([lr_sklearn.intercept_, np.squeeze(lr_sklearn.coef_)])
    else:
        weights_sklearn = np.squeeze(lr_sklearn.coef_)
    np.testing.assert_allclose(weights_numpy, weights_sklearn)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_iris_data(california_housing_dataset: DatasetTrainTest, fit_intercept: bool) -> None:
    X_train, y_train, _, _ = california_housing_dataset
    lr_sklearn = LinearRegressionSklearn(fit_intercept=fit_intercept)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit(X_train, y_train)
    lr_sklearn.fit(X_train, y_train)
    assert lr_numpy.weights is not None
    weights_numpy = np.squeeze(lr_numpy.weights)
    if fit_intercept:
        weights_sklearn = np.concatenate([[lr_sklearn.intercept_], lr_sklearn.coef_])
    else:
        weights_sklearn = np.squeeze(lr_sklearn.coef_)
    np.testing.assert_allclose(weights_numpy, weights_sklearn)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_raises_on_mismatched_shapes(fit_intercept: bool) -> None:
    np.random.seed(42)
    X_train = np.random.randn(10, 3)
    y_train = np.random.randn(5, 1)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="Dimensions do not match"):
        lr_numpy.fit(X_train, y_train)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_raises_on_rank_deficient_matrix(fit_intercept: bool) -> None:
    np.random.seed(42)
    X_train = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 4]])
    y_train = np.array([0, 1, 2])
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="Singular matrix"):
        lr_numpy.fit(X_train, y_train)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_and_animate_raises_on_mismatched_shapes(fit_intercept: bool) -> None:
    np.random.seed(42)
    X_train = np.random.randn(10, 3)
    y_train = np.random.randn(5, 1)
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="Dimensions do not match"):
        lr_numpy.fit_and_animate(X_train, y_train)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_and_animate_raises_on_more_than_one_feature(fit_intercept: bool) -> None:
    np.random.seed(42)
    X_train = np.random.randn(3, 2)
    y_train = 2 * X_train + 1
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with pytest.raises(ValueError, match="Animation only supports one feature for visualization"):
        lr_numpy.fit_and_animate(X_train, y_train)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_and_animate_calls_render(fit_intercept: bool) -> None:
    np.random.seed(42)
    X_train = np.random.randn(8, 1)
    y_train = 2 * X_train + 1
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    with patch("manim.Scene.render") as mock_render:
        lr_numpy.fit_and_animate(X_train, y_train, quality="low_quality")
        mock_render.assert_called_once()


@pytest.mark.slow
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_and_animate_create_video(tmp_path: Path, fit_intercept: bool) -> None:
    np.random.seed(42)
    output_dir = tmp_path / "video_out"
    X_train = np.random.randn(8, 1)
    y_train = 2 * X_train + 1
    lr_numpy = LinearRegressionNumpy(fit_intercept=fit_intercept)
    lr_numpy.fit_and_animate(X_train, y_train, "low_quality", str(output_dir))
    assert True
