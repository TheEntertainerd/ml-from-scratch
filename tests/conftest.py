import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pytest
from typing import cast
from tests.types import PolynomialCase, DatasetTrainTest

@pytest.fixture(scope="module")
def synthetic_data() -> DatasetTrainTest:
    np.random.seed(42)
    shape = (10000,5)
    X_train = np.random.randint(0,100,shape).astype(np.float64)
    y_train = X_train[:,0] * 3 + X_train[:,1] * 5 + 10 + np.random.random((shape[0]))*5 
    X_test = np.random.randint(0,100,shape).astype(np.float64)   
    y_test = X_test[:,0] * 3 + X_test[:,1] * 5 + 10
    return X_train, y_train.reshape(-1,1), X_test, y_test.reshape(-1,1)


@pytest.fixture(scope="module")
def california_housing_dataset() -> DatasetTrainTest:
    np.random.seed(42)
    california = fetch_california_housing()
    X = california.data
    y = california.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 
    )
    return X_train, y_train, X_test, y_test


@pytest.fixture(params=[
    # Constant polynomial (2 vars)
    (
        {(0, 0): 8.0},
        np.array([2.0, 4.0]),
        8.0,
        np.array([0.0, 0.0])
    ),
    # 2 variables
    (
        {(2, 0): 2.0, (1, 1): 1.0, (0, 2): 3.0},   # 2x² + xy + 3y²
        np.array([2.0, 4.0]),
        64.0,
        np.array([12.0, 26.0])
    ),
    # 3 variables
    (
        {(2, 0, 0): 2.0, (1, 1, 1): 1.0, (0, 1, 1): 3.0, (0, 0, 2): -1.0},
        np.array([1.0, -2.0, 3.0]),
        -31.0,
        np.array([-2.0, 12.0, -14.0])
    ),
    # 4 variables
    (
        {(3, 0, 0, 0): 1.0, (1, 1, 0, 1): 1.0, (0, 1, 1, 0): -4.0,
         (0, 0, 3, 0): -1.0, (0, 0, 0, 2): 2.0},
        np.array([2.0, -1.0, 0.5, 3.0]),
        21.875,
        np.array([9.0, 4.0, 3.25, 10.0])
    ),
    # 5 variables
    (
        {(1, 1, 0, 0, 0): 1.0, (2, 0, 1, 1, 0): 1.0, (0, 1, 0, 1, 1): -1.0,
         (0, 0, 2, 1, 0): 2.0, (0, 0, 0, 0, 1): -3.0},
        np.array([1.0, 2.0, -1.0, 0.5, 3.0]),
        -9.5,
        np.array([1.0, -0.5, -1.5, -5.0, -4.0])
    ),
    # 6 variables
    (
        {(2, 0, 1, 0, 0, 0): 1.0, (1, 1, 0, 0, 0, 0): 3.0, (0, 1, 0, 0, 0, 1): -1.0,
         (0, 0, 2, 0, 0, 0): -2.0, (0, 0, 0, 1, 1, 1): 1.0, (0, 0, 0, 0, 3, 0): 1.0},
        np.array([1.0, -2.0, 0.5, 2.0, -1.0, 3.0]),
        -7.0,
        np.array([-5.0, 0.0, -1.0, -3.0, 9.0, 0.0])
    ),
    # 8 variables
    (
        {(1, 1, 0, 0, 0, 0, 0, 0): 1.0, (1, 0, 1, 0, 1, 0, 0, 0): 1.0,
         (0, 1, 0, 1, 0, 1, 0, 0): -1.0, (0, 0, 1, 1, 0, 0, 0, 0): -1.0,
         (0, 0, 0, 0, 2, 0, 0, 0): 1.0, (0, 0, 0, 0, 0, 1, 1, 1): -2.0,
         (0, 0, 0, 0, 0, 0, 3, 0): 1.0},
        np.array([1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, -1.0]),
        7.875,
        np.array([-1.0, 2.0, 2.5, 5.0, 5.0, 2.0, 2.75, 6.0])
    ),
])
def polynomial_case(request: pytest.FixtureRequest) -> PolynomialCase:
    return cast(PolynomialCase, request.param)

