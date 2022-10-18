import glob
from typing import List

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


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
    Solves the system of homogenous equations using eigen-decomposition.
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
    Returns the extrinsic rotation matrices and translation vectors.
    """
    A_inv = np.linalg.inv(camera_matrix)

    R_mats, t_vecs = [], []
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

        R_mats.append(np.array([r1, r2, r3]).T)
        t_vecs.append(np.array([t]).T)

    return np.array(R_mats), np.array(t_vecs)


def get_projected_corners(camera_matrix, r_mats, t_vecs, world_points):
    """
    :param camera_matrix:(3x3)
    :param r_mats: numpy array of all rotation matrices (mx3x3)
    :param t_vecs: numpy array of all translational vectors (mx3x1)
    :param world_points: numpy array of all world points (mx3)
    :return: image_points: 13x3x54
    """
    # Homogenize world points
    world_points = world_points.T  # 3xm
    corners_count = world_points.shape[1]
    ones = np.ones((1, corners_count))
    world_points = np.append(world_points, ones, axis=0)

    # Get extrinsic matrix from r_mat and t_vec
    extrinsic_matrices = np.append(r_mats, t_vecs, axis=2)

    # Get projected coords
    img_coords = np.matmul(
        camera_matrix,
        np.matmul(extrinsic_matrices, world_points)
    )

    # homogenize
    img_coords[:, 0, :] = img_coords[:, 0, :] / img_coords[:, 2, :]
    img_coords[:, 1, :] = img_coords[:, 1, :] / img_coords[:, 2, :]
    img_coords[:, 2, :] = img_coords[:, 2, :] / img_coords[:, 2, :]

    return img_coords


def objective_function(x, detected_corners, world_points):
    """
    :param x: param vector encoding camera matrix, rotation & translation vec.
    :param detected_corners: 13
    :param world_points:
    :return:
    """
    # TODO: Make types of img and world points the same

    # print("det_cor", detected_corners.shape)
    # print("wor_pts", world_points.shape)

    detected_corners = np.array(detected_corners)
    detected_corners = detected_corners.reshape((13, 54, -1))
    detected_corners = detected_corners.swapaxes(1, 2)

    ones = np.ones((13, 1, 54))
    detected_corners = np.concatenate((detected_corners, ones), axis=1)

    camera_matrix, Rs, ts = from_parameter_vector(x)
    projected_corners = get_projected_corners(camera_matrix, Rs, ts, world_points)

    residual = detected_corners - projected_corners

    residual = residual.swapaxes(1, 2)

    sum = 0
    for i in range(13):
        for j in range(54):
            sum += residual[i, j, 0]**2 + residual[i, j, 1]**2 + residual[i, j, 2]**2
    return sum


def to_parameter_vector(
        camera_matrix: np.array(List[List[float]]),
        r_mats: np.array(List[List[float]]),
        t_vecs: np.array(List[List[float]])
    ) -> np.array:
    """
    Appends all the parameters to be optimized into a vector to be passed to the
    optimization function.
    :return: 5 + 13*6: 83 (dim: 1x83)
    """
    parameter_vector = []

    # Append intrinsics
    parameter_vector.extend(
        [
            camera_matrix[0, 0],
            camera_matrix[0, 1],
            camera_matrix[0, 2],
            camera_matrix[1, 1],
            camera_matrix[1, 2],
        ]
    )

    # Append parametrized R
    r_params = []
    for r_mat in r_mats:
        r = R.from_matrix(r_mat)
        r_parametrised = r.as_mrp()
        r_params.extend(r_parametrised)
    parameter_vector.extend(r_params)

    # Append t_vecs
    t_params = np.array([])
    for t_vec in t_vecs:
        t_vec = t_vec.flatten()
        t_params = np.append(t_params, t_vec, axis=0)
    parameter_vector.extend(list(t_params))

    return np.array(parameter_vector)


def from_parameter_vector(vec: np.array):
    """
    Breaks down the parameter vector into its foundational elements and returns
    the camera matrix, rotation matrix and the translation vector
    """
    camera_matrix = np.array(
        [
            [vec[0], vec[1], vec[2]],
            [0, vec[3], vec[4]],
            [0, 0, 1]
        ]
    )

    Rs = []
    r_params = np.array(vec[5:44]).reshape((-1, 3))  # 13x3
    for r_param in r_params:
        r = R.from_mrp(r_param)
        Rs.append(r.as_matrix())

    ts = []
    t_params = np.array(vec[44:83]).reshape((-1, 3))
    for t_param in t_params:
        ts.append(np.array([t_param]).T)

    return camera_matrix, Rs, ts


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

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray, (9, 6), None)
        if ret:
            all_world_points.append(world_pts)
            corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
            all_img_points.append(corners)

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
    r_mats, t_vecs = get_extrinsics(mat, homographies)

    param_vector = to_parameter_vector(mat, r_mats, t_vecs)

    res = minimize(
        objective_function,
        param_vector,
        args=(all_img_points, world_pts),
    )

    optimized_param_vec = res.x
    opt_cam_mat, opt_Rs, opt_ts = from_parameter_vector(optimized_param_vec)


if __name__ == '__main__':
    main()
