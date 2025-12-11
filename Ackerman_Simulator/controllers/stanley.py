import numpy as np

class StanleyController:
    def __init__(self, k_gain=0.5, k_soft=1.0, error_recovery=True):
        self.k = k_gain      # Control gain (determines how aggressive it corrects error)
        self.k_soft = k_soft # Softening parameter to prevent division by zero at low speed
        self.error_recovery = error_recovery  # Enable aggressive recovery when far from path

    def compute(self, vehicle_state, path_x, path_y, target_idx):
        """
        Calculates steering angle using Stanley method.
        Stanley method: delta = theta_e + arctan(k * e / v)
        where theta_e is heading error and e is cross-track error
        
        Enhanced with error recovery: When cross-track error grows large,
        increase steering gain to recover faster.
        """
        x = vehicle_state[0]
        y = vehicle_state[1]
        yaw = vehicle_state[2]
        v = vehicle_state[3]  # Longitudinal velocity

        # Import vehicle parameters
        from models.vehicle_dynamics import VehicleParams
        
        # Calculate front axle position (Stanley uses front axle reference point)
        front_axle_x = x + VehicleParams.CG_TO_FRONT * np.cos(yaw)
        front_axle_y = y + VehicleParams.CG_TO_FRONT * np.sin(yaw)

        # 1. Find the nearest path point to the front axle
        # This ensures we're measuring error from the closest point on path
        distances = np.hypot(front_axle_x - path_x, front_axle_y - path_y)
        nearest_idx = np.argmin(distances)
        
        # Use nearest point for cross-track error, but target_idx for heading
        # This prevents the error from growing when looking ahead

        # 2. Calculate path heading at nearest point
        if nearest_idx < len(path_x) - 1:
            dx = path_x[nearest_idx + 1] - path_x[nearest_idx]
            dy = path_y[nearest_idx + 1] - path_y[nearest_idx]
        else:
            dx = path_x[nearest_idx] - path_x[nearest_idx - 1]
            dy = path_y[nearest_idx] - path_y[nearest_idx - 1]
        
        path_yaw = np.arctan2(dy, dx)
        
        # 3. Calculate heading error (theta_e)
        heading_error = path_yaw - yaw
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # 4. Calculate cross-track error (e)
        # Vector from nearest path point to front axle
        dx_error = front_axle_x - path_x[nearest_idx]
        dy_error = front_axle_y - path_y[nearest_idx]
        
        # Project error onto path normal (perpendicular to path direction)
        # Positive error = vehicle is to the left of path
        # Use cross product to determine sign: (path_vec) x (error_vec)
        cross_track_error = dx * dy_error - dy * dx_error
        # Normalize by path segment length to get signed distance
        path_length = np.hypot(dx, dy)
        if path_length > 1e-6:
            cross_track_error = cross_track_error / path_length

        # 5. Adaptive gain based on error magnitude (error recovery)
        # When far from path (|e| > 3m), increase k gain for faster recovery
        k_adaptive = self.k
        if self.error_recovery:
            abs_error = abs(cross_track_error)
            if abs_error > 3.0:
                # Scale gain up to 2x for large errors
                recovery_factor = 1.0 + (abs_error - 3.0) / 10.0
                k_adaptive = self.k * np.clip(recovery_factor, 1.0, 2.0)

        # 6. Stanley control law
        # delta = theta_e + arctan(k * e / (v + k_soft))
        cross_track_term = np.arctan2(k_adaptive * cross_track_error, (abs(v) + self.k_soft))
        
        steer_cmd = heading_error + cross_track_term
        
        return steer_cmd, cross_track_error

class PIDController:
    def __init__(self, Kp=0.5, Ki=0.01, Kd=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative