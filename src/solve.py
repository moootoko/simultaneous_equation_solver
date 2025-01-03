import numpy as np


def solve_linear_equations(
    a11: float, a12: float, b1: float, a21: float, a22: float, b2: float
):
    """
    2x2の連立一次方程式を解く関数。

    方程式の形式: \\
    a11 * x + a12 * y = b1 \\
    a21 * x + a22 * y = b2

    戻り値: \\
    numpy.ndarray 解が存在する場合は解の配列を返す

    例外:\\
    解が存在しない場合はValueErrorを返す。
    """

    # 係数行列
    A = np.array([[a11, a12], [a21, a22]])
    # 定数ベクトル
    B = np.array([b1, b2])

    # 係数行列の行列式を計算する
    det = np.linalg.det(A)

    if det == 0:
        raise ValueError("連立方程式の解が存在しません。")
    else:
        # 連立方程式を解く
        solution = np.linalg.solve(A, B)
        return solution
