import pytest
import numpy as np
from solve import solve_linear_equations


@pytest.mark.parametrize(
    "a11, a12, b1, a21, a22, b2, x1, x2",
    [
        (1, 2, 3, 4, 5, 6, -1, 2),
        (3, 3, 6, 5, 0, 5, 1, 1),
        (2, 1, 6, -2, 2, 12, 0, 6),
    ],
)
def test_solve_linear_equations(a11, a12, b1, a21, a22, b2, x1, x2):
    result = solve_linear_equations(a11, a12, b1, a21, a22, b2)

    assert np.allclose(result, np.array([x1, x2]))


@pytest.mark.NoSolution
@pytest.mark.parametrize(
    "a11, a12, b1, a21, a22, b2",
    [
        (2, 1, 0, 4, 2, 12),
        (-3, 2, -3, 9, -6, 5),
        (5, -2, 1, -10, 4, 3),
    ],
)
def test_solve_linear_equations_no_solution(a11, a12, b1, a21, a22, b2):
    with pytest.raises(ValueError, match="連立方程式の解が存在しません。"):
        solve_linear_equations(a11, a12, b1, a21, a22, b2)
