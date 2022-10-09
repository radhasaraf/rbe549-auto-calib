from typing import List
import numpy as np


def get_v_ij_from_homography(h: List[List[float]], i: int, j: int) -> List:
    """
    Returns the v_ij vector required for setting up the system of homogenous
    linear equations.
    """
    i -= 1
    j -= 1
    return [
        h[i][0] * h[j][0],
        h[i][0] * h[j][1] + h[i][1] * h[j][0],
        h[i][1] * h[j][1],
        h[i][2] * h[j][0] + h[i][0] * h[j][2],
        h[i][2] * h[j][1] + h[i][1] * h[j][2],
        h[i][2] * h[j][2],
    ]


def get_intrinsic_mat(b_vec: List) -> List[List[float]]:
    """
    Calculates the intrinsic matrix from the b vector according to Appendix A
    in the reference paper.
    """
    b11 = b_vec[0]
    b12 = b_vec[1]
    b22 = b_vec[2]
    b13 = b_vec[3]
    b23 = b_vec[4]
    b33 = b_vec[5]

    v = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 ** 2)
    lmda = b33 - [b13 ** 2 + v * (b12 * b13 - b11 * b23)] / b11
    alpha = np.sqrt(lmda / b11)
    beta = np.sqrt(lmda * b11 / (b11 * b22 - b12 ** 2))
    gamma = -b12 * (alpha ** 2) * beta / lmda
    u = gamma * v / beta - b13 * (alpha ** 2) / lmda

    return [
        [alpha, gamma, u],
        [0,     beta,  v],
        [0,     0,     1]
    ]
