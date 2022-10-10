import glob
from typing import List

import cv2
import numpy as np


def get_v_ij(h: np.array(List[List[float]]), i: int, j: int) -> np.array(List[float]):
    """
    Returns the v_ij vector required for setting up the system of homogenous
    linear equations.
    """
    h = np.transpose(h)
    i -= 1
    j -= 1
    return np.array([
        h[i][0] * h[j][0],
        h[i][0] * h[j][1] + h[i][1] * h[j][0],
        h[i][1] * h[j][1],
        h[i][2] * h[j][0] + h[i][0] * h[j][2],
        h[i][2] * h[j][1] + h[i][1] * h[j][2],
        h[i][2] * h[j][2],
    ])


def solve_homogenous_sys(parameter_mat: np.array(List[List[float]])) -> np.array(List[float]):
    """

    """
    e_vals, e_rows = np.linalg.eig(parameter_mat.T @ parameter_mat)
    e_vecs = e_rows.T  # Take transpose because vectors are columns, not rows
    return e_vecs[np.argmin(e_vals)]


def get_intrinsic_mat(b_vec: np.array(List[float])) -> np.array(List[List[float]]):
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
    lmda = b33 - ((b13 ** 2) + (v * (b12 * b13 - b11 * b23))) / b11
    alpha = np.sqrt(lmda / b11)
    beta = np.sqrt(lmda * b11 / ((b11 * b22) - (b12 ** 2)))
    gamma = -b12 * (alpha ** 2) * beta / lmda
    u = gamma * v / beta - b13 * (alpha ** 2) / lmda

    return np.array([
        [alpha, gamma, u],
        [0,     beta,  v],
        [0,     0,     1]
    ])


def get_extrinsics(camera_matrix: np.array(List[List[float]]), homographies: List[List[List[float]]]):
    """

    """
    A_inv = np.linalg.inv(camera_matrix)

    all_extrinsics = []
    for h in homographies:

        # Get column vectors of H
        h_T = h.T
        h1 = h_T[0].T
        h2 = h_T[1].T
        h3 = h_T[2].T

        r1 = A_inv @ h1
        r2 = A_inv @ h2

        lmda1 = 1 / np.linalg.norm(r1)
        lmda2 = 1 / np.linalg.norm(r2)

        r1 *= lmda1
        r2 *= lmda2

        r3 = np.cross(r1, r2)
        t = lmda1 * (A_inv @ h3)  # Any lmda is fine since both are close

        all_extrinsics.append([r1, r2, r3, t])
    return all_extrinsics


def main():

    # termination criteria for sub pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare world points: (0,0,0), (1,0,0), ....,(8,5,0)
    # unit coz actual length doesn't make a diff since homography is to scale
    world_pts = np.zeros((9 * 6, 3), np.float32)
    world_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store world points and image points from all the images.
    all_world_points, all_img_points = [], []

    calib_image_files = glob.glob('./Calibration_Imgs/*.jpg')
    for file in calib_image_files:
        img = cv2.imread(file)
        # img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[0] / 3)))

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray, (9, 6), None)
        if ret:
            all_world_points.append(world_pts)
            corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            all_img_points.append(corners)

        # # Draw and display the corners
        # cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(1000)

    # Form homogenous system of linear equations:
    V = []
    homographies = []
    for i in range(len(all_world_points)):
        h, _ = cv2.findHomography(all_world_points[i], all_img_points[i], cv2.RANSAC)

        # Save homographies for optimization
        homographies.append(h)

        equ1 = get_v_ij(h, 1, 2)
        equ2 = np.subtract(get_v_ij(h, 1, 1), get_v_ij(h, 2, 2))
        V.extend([equ1, equ2])

    # Find solution to homogenous system
    V = np.array(V)
    b = solve_homogenous_sys(V)

    # Get camera matrix
    mat = get_intrinsic_mat(b)
    print("Camera matrix: ", mat)

    extrs = get_extrinsics(mat, homographies)
    print("Extrinsics(r1, r2, r3, t): ", extrs[0])


if __name__ == '__main__':
    main()
