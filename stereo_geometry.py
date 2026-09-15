"""Explicit Scene11/real stereo profiles; no object ground truth is used.

All output points use OpenCV camera axes and metres. Input correspondences
must identify the SAME physical point in synchronized, unrectified images.
This module does not find correspondences or control the robot.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


DEFAULT_CALIBRATION = Path(__file__).with_name('g1_head_camera_calibrations.yaml')
CV_TO_GL = np.diag([1., -1., -1., 1.])


@dataclass
class StereoGeometry:
    mode: str
    image_size_wh: tuple
    K_left: np.ndarray
    K_right: np.ndarray
    D_left: np.ndarray
    D_right: np.ndarray
    R: np.ndarray  # left OpenCV -> right OpenCV
    T: np.ndarray  # metres
    T_head_left_cv: object = None

    @classmethod
    def load(cls, mode, path=DEFAULT_CALIBRATION):
        if mode not in ('sim', 'real'):
            raise ValueError('Choose mode explicitly: sim or real')
        with open(path) as stream:
            data = yaml.safe_load(stream)['stereo_head_camera']
        size = tuple(data['image_size_wh'])
        cameras = []
        mounts = []
        for side in ('left', 'right'):
            intr = data[side]['intrinsics']
            K = np.array(intr['K'], dtype=float)
            D = np.array(intr['distortion_coefficients_k1_k2_p1_p2_k3'], dtype=float)
            if mode == 'sim':
                # Current Isaac renderer: square pixels, centered principal point.
                K = np.array([[K[0, 0], 0, size[0] / 2],
                              [0, K[0, 0], size[1] / 2], [0, 0, 1.]])
                D = np.zeros(5)
                mount = data[side]['isaac_render_offset']
                q = mount['rotation_quaternion_wxyz']
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_quat([*q[1:], q[0]]).as_matrix()
                transform[:3, 3] = mount['translation_m']
                # Offset is already included in the supplied rendered mount.
                mounts.append(transform @ CV_TO_GL)
            cameras.append((K, D))
        if mode == 'sim':
            relative = np.linalg.inv(mounts[1]) @ mounts[0]
            R, T = relative[:3, :3], relative[:3, 3]
        else:
            relative = data['stereo_left_to_right']
            R = np.array(relative['R'], dtype=float)
            T = np.array(relative['T_mm'], dtype=float) / 1000
        return cls(mode, size, cameras[0][0], cameras[1][0],
                   cameras[0][1], cameras[1][1], R, T,
                   mounts[0] if mounts else None)

    def triangulate(self, left_px, right_px, *, image_size_wh,
                    max_reprojection_px=2.0):
        """Return Nx3 left-camera points; reject degenerate/inconsistent pairs.

        Reprojection consistency cannot reject every wrong match. Appearance
        matching and task-specific depth/workspace checks are still required.
        Rectified pixels must NOT be supplied to this raw-image interface.
        """
        if tuple(image_size_wh) != self.image_size_wh:
            raise ValueError('Image resolution differs from calibration; do not silently resize')
        left, right = np.asarray(left_px, float), np.asarray(right_px, float)
        if (left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape
                or len(left) == 0 or not np.isfinite([left, right]).all()):
            raise ValueError('Expected finite matching Nx2 pixel arrays')
        if not np.isfinite(max_reprojection_px) or max_reprojection_px <= 0:
            raise ValueError('Reprojection threshold must be positive and finite')
        normalized = [cv2.undistortPoints(p[:, None, :], K, D).reshape(-1, 2)
                      for p, K, D in ((left, self.K_left, self.D_left),
                                      (right, self.K_right, self.D_right))]
        h = cv2.triangulatePoints(np.eye(3, 4), np.column_stack((self.R, self.T)),
                                 normalized[0].T, normalized[1].T)
        if not np.isfinite(h).all() or np.any(np.abs(h[3]) < 1e-10):
            raise ValueError('Degenerate triangulation')
        xyz = (h[:3] / h[3]).T
        right_xyz = xyz @ self.R.T + self.T
        if np.any(xyz[:, 2] <= 0) or np.any(right_xyz[:, 2] <= 0):
            raise ValueError('Triangulated point is behind a camera')
        for points, observed, K, D in ((xyz, left, self.K_left, self.D_left),
                                       (right_xyz, right, self.K_right, self.D_right)):
            projected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
            if np.any(np.linalg.norm(projected - observed, axis=1) > max_reprojection_px):
                raise ValueError('Stereo correspondence exceeds reprojection threshold')
        return xyz
