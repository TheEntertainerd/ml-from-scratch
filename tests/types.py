from typing import Tuple, Dict
import numpy as np
import numpy.typing as npt


PolynomialCase = Tuple[Dict[Tuple[int, ...], float], np.ndarray, float, np.ndarray]

DatasetTrainTest = tuple[
    npt.NDArray[np.float64],  # X_train
    npt.NDArray[np.float64],  # y_train
    npt.NDArray[np.float64],  # X_test
    npt.NDArray[np.float64],  # y_test
]
