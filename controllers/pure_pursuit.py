"""Pure Pursuit path-tracking controller.

Simple geometric controller: steer to the lookahead target point on the path.
Stable and robust; good fallback when LQR tuning is problematic.
"""

import numpy as np
from models.vehicle_dynamics import VehicleParams


class PurePursuitController:
    def __init__(self, lookahead_base=3.0, lookahead_gain=0.2, min_lookahead=2.0, max_lookahead=8.0):
        self.lookahead_base = lookahead_base
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead

    def _find_target(self, x, y, path_x, path_y, lookahead):
        d = np.hypot(path_x - x, path_y - y)
        idx = np.argmin(d)
        for i in range(idx, len(path_x)):
            if np.hypot(path_x[i] - x, path_y[i] - y) >= lookahead:
                return i
        return len(path_x) - 1

    def compute(self, vehicle_state, path_x, path_y, target_idx):
        x, y, yaw, vx = vehicle_state[:4]
        v = max(vx, 0.0)

        lookahead = self.lookahead_base + self.lookahead_gain * v
        lookahead = np.clip(lookahead, self.min_lookahead, self.max_lookahead)

        idx = self._find_target(x, y, path_x, path_y, lookahead)
        tx, ty = path_x[idx], path_y[idx]

        # Transform target into vehicle frame
        dx = tx - x
        dy = ty - y
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = -np.sin(yaw) * dx + np.cos(yaw) * dy

        if abs(local_x) < 1e-6:
            return 0.0, 0.0

        curvature = 2.0 * local_y / (lookahead ** 2)
        steer = np.arctan(curvature * VehicleParams.WHEELBASE)
        steer = np.clip(steer, -VehicleParams.MAX_STEER, VehicleParams.MAX_STEER)

        # Cross-track error (signed) using path nearest point
        nearest = np.argmin(np.hypot(path_x - x, path_y - y))
        if nearest < len(path_x) - 1:
            dxp = path_x[nearest + 1] - path_x[nearest]
            dyp = path_y[nearest + 1] - path_y[nearest]
        else:
            dxp = path_x[nearest] - path_x[nearest - 1]
            dyp = path_y[nearest] - path_y[nearest - 1]
        heading = np.arctan2(dyp, dxp)
        perp = np.array([-np.sin(heading), np.cos(heading)])
        cte = -np.dot(np.array([x - path_x[nearest], y - path_y[nearest]]), perp)

        return steer, cte


__all__ = ["PurePursuitController"]