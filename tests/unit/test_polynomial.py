import numpy as np
from src.utils.polynomial import MultivariatePolynomial
import pytest
from tests.types import PolynomialCase


def test_init_raises_on_empty_dictionary() -> None:
    polynome_dictionary: dict[tuple[int, ...], float] = {}
    with pytest.raises(ValueError, match="Dictionary is empty"):
        _ = MultivariatePolynomial(polynome_dictionary)


def test_init_raises_on_key_not_tuple() -> None:
    polynome_dictionary = {
        2: 2,
    }
    with pytest.raises(ValueError, match="Key must be tuple"):
        _ = MultivariatePolynomial(polynome_dictionary)  # type: ignore[arg-type]


def test_init_raises_on_tuples_different_length() -> None:
    polynome_dictionary: dict[tuple[int, ...], float] = {
        (1, 0, 0): 2,
        (1, 2): 2,
    }
    with pytest.raises(ValueError, match="All tuples must have same length"):
        _ = MultivariatePolynomial(polynome_dictionary)


def test_call_raises_on_mismatch_between_polynome_and_parameter() -> None:
    polynome_dictionary: dict[tuple[int, ...], float] = {
        (2, 0): 2,  # 2x²
        (0, 2): 1,  # y²
    }
    initial_parameters = np.array([2.0, 2.0, 4.0])
    polynome = MultivariatePolynomial(polynome_dictionary)
    with pytest.raises(ValueError, match="Number of variables passed does not match that of the polynome"):
        polynome(initial_parameters)


def test_call(polynomial_case: PolynomialCase) -> None:
    polynome_dictionary, initial_parameters, expected_value, expected_grad = polynomial_case
    polynome = MultivariatePolynomial(polynome_dictionary)
    np.testing.assert_allclose(polynome(initial_parameters), expected_value)


def test_get_derivative_raises_on_mismatch_between_polynome_and_parameter() -> None:
    polynome_dictionary: dict[tuple[int, ...], float] = {
        (2, 0): 2,  # 2x²
        (0, 2): 1,  # y²
    }
    initial_parameters = np.array([2.0, 2.0, 4.0])
    polynome = MultivariatePolynomial(polynome_dictionary)
    with pytest.raises(ValueError, match="Number of variables passed does not match that of the polynome"):
        polynome.get_derivative(initial_parameters, 1)


def test_get_derivative(polynomial_case: PolynomialCase) -> None:
    polynome_dictionary, initial_parameters, expected_value, expected_grad = polynomial_case
    polynome = MultivariatePolynomial(polynome_dictionary)
    for i in range(len(expected_grad)):
        np.testing.assert_allclose(polynome.get_derivative(initial_parameters, i), expected_grad[i])


def test_get_gradient(polynomial_case: PolynomialCase) -> None:
    polynome_dictionary, initial_parameters, expected_value, expected_grad = polynomial_case
    polynome = MultivariatePolynomial(polynome_dictionary)
    print(polynome.get_gradient(initial_parameters))
    np.testing.assert_allclose(polynome.get_gradient(initial_parameters), expected_grad)
