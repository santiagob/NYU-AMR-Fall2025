"""LQR path-tracking controller for bicycle models.

Reference: Based on the standard lateral LQR steering controller from
PythonRobotics (Atsushi Sakai et al.). Uses a linearized bicycle model
discretized at the current forward speed.

State vector: [e_y, e_y_dot, e_yaw, yaw_rate]
Control: steering angle (delta)

This controller is more "state of the art" than pure Stanley/Pure Pursuit
in that it optimizes a quadratic cost over the predicted lateral/yaw error
dynamics each step (receding horizon). For speed, pair with your existing
PID.
"""

import numpy as np
from models.vehicle_dynamics import VehicleParams


class LQRController:
    def __init__(self, q_weights=None, r_weight=1.0, dt=0.1,
                 cte_stabilizer=0.15, soft_speed=1.0):
        # Weights for [e_y, e_y_dot, e_yaw, yaw_rate]
        self.Q = np.diag(q_weights if q_weights is not None else [2.5, 0.6, 4.0, 0.6])
        self.R = np.array([[r_weight]])
        self.dt = dt
        # Small Stanley-like stabilizer to damp cross-track flipping
        self.cte_stabilizer = cte_stabilizer
        self.soft_speed = soft_speed

    def _nearest_index(self, x, y, path_x, path_y):
        d = np.hypot(path_x - x, path_y - y)
        idx = int(np.argmin(d))
        return idx

    def _calc_path_yaw(self, path_x, path_y, idx):
        if idx < len(path_x) - 1:
            dx = path_x[idx + 1] - path_x[idx]
            dy = path_y[idx + 1] - path_y[idx]
        else:
            dx = path_x[idx] - path_x[idx - 1]
            dy = path_y[idx] - path_y[idx - 1]
        return np.arctan2(dy, dx)

    def _calc_path_curvature(self, path_x, path_y, idx):
        # Use centered finite difference when possible
        if idx == 0:
            idxs = [idx, idx + 1, idx + 2] if len(path_x) > 2 else [0, 1, 1]
        elif idx >= len(path_x) - 1:
            idxs = [idx - 2, idx - 1, idx]
        else:
            idxs = [idx - 1, idx, idx + 1]

        x0, x1, x2 = path_x[idxs[0]], path_x[idxs[1]], path_x[idxs[2]]
        y0, y1, y2 = path_y[idxs[0]], path_y[idxs[1]], path_y[idxs[2]]

        # Compute curvature kappa
        a = np.hypot(x1 - x0, y1 - y0)
        b = np.hypot(x2 - x1, y2 - y1)
        c = np.hypot(x2 - x0, y2 - y0)
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 0.0)
        area = np.sqrt(area)
        kappa = (4.0 * area) / (a * b * c)
        # Sign from cross product
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if cross < 0:
            kappa = -kappa
        return kappa

    def _solve_dare(self, A, B):
        """Solve discrete algebraic Riccati equation via iteration."""
        P = self.Q.copy()
        for _ in range(50):
            P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(self.R + B.T @ P @ B) @ B.T @ P @ A + self.Q
            if np.max(np.abs(P_next - P)) < 1e-6:
                break
            P = P_next
        return P

    def _lqr_gain(self, A, B):
        P = self._solve_dare(A, B)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        return K

    def compute(self, vehicle_state, path_x, path_y, target_idx):
        # Accept 4+ state entries: [x, y, yaw, vx, vy?, r?]
        x, y, yaw, vx = vehicle_state[:4]
        v = max(vx, 0.1)  # avoid divide by zero

        # Find nearest path point (ignore provided idx to stay robust)
        nearest_idx = self._nearest_index(x, y, path_x, path_y)
        ref_yaw = self._calc_path_yaw(path_x, path_y, nearest_idx)
        kappa = self._calc_path_curvature(path_x, path_y, nearest_idx)

        # Cross-track error (signed, using path normal)
        dx = path_x[nearest_idx] - x
        dy = path_y[nearest_idx] - y
        # path normal (perp to heading)
        perp = np.array([-np.sin(ref_yaw), np.cos(ref_yaw)])
        # Flip sign so positive error = vehicle is to the left of path (consistent with Stanley logic)
        e_y = -np.dot([dx, dy], perp)

        # Heading error
        e_yaw = (ref_yaw - yaw + np.pi) % (2 * np.pi) - np.pi

        # Lateral error rate (approx.)
        e_y_dot = v * np.sin(e_yaw)

        # Yaw rate (from vehicle state if available)
        yaw_rate = 0.0
        if len(vehicle_state) >= 6:
            yaw_rate = vehicle_state[5]

        # Linearized bicycle model (discrete)
        L = VehicleParams.WHEELBASE
        A = np.eye(4)
        A[0, 1] = self.dt
        A[0, 2] = v * self.dt
        A[1, 2] = v
        A[2, 3] = self.dt

        B = np.zeros((4, 1))
        B[1, 0] = v / L
        B[3, 0] = v / L
        B = B * self.dt

        K = self._lqr_gain(A, B)

        state_vec = np.array([[e_y], [e_y_dot], [e_yaw], [yaw_rate]])
        steer_fb = float(-K @ state_vec)

        # Feedforward term based on path curvature: delta_ff = arctan(L * kappa)
        L = VehicleParams.WHEELBASE
        steer_ff = np.arctan2(L * kappa, 1.0)

        # Stanley-like damping on cross-track to prevent sign flips at tight turns
        cte_term = np.arctan2(self.cte_stabilizer * e_y, (abs(v) + self.soft_speed))

        steer_cmd = steer_fb + steer_ff + cte_term

        # Clip to vehicle limits
        steer_cmd = np.clip(steer_cmd, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)

        return steer_cmd, e_y


__all__ = ["LQRController"]