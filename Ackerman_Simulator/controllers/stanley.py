import numpy as np

class StanleyController:
    def __init__(self, k_gain=0.5, k_soft=1.0):
        self.k = k_gain      # Control gain (determines how aggressive it corrects error)
        self.k_soft = k_soft # Softening parameter to prevent division by zero at low speed

    def compute(self, vehicle_state, path_x, path_y, target_idx):
        """
        Calculates steering angle.
        """
        x = vehicle_state[0]
        y = vehicle_state[1]
        yaw = vehicle_state[2]
        v = vehicle_state[3] # Longitudinal velocity

        # 1. Calculate Heading Error (Theta_e)
        # Find tangent of path at target index
        if target_idx < len(path_x) - 1:
            dx = path_x[target_idx+1] - path_x[target_idx]
            dy = path_y[target_idx+1] - path_y[target_idx]
        else:
            dx = path_x[target_idx] - path_x[target_idx-1]
            dy = path_y[target_idx] - path_y[target_idx-1]
        
        path_yaw = np.arctan2(dy, dx)
        heading_error = path_yaw - yaw

        # Normalize angle to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        # 2. Calculate Cross-Track Error (e)
        # Distance from front axle to path
        # Note: Stanley is defined for the Front Axle
        front_axle_x = x + 1.2 * np.cos(yaw) # 1.2 is CG_TO_FRONT
        front_axle_y = y + 1.2 * np.sin(yaw)
        
        # Calculate distance to closest point on line defined by target_idx
        # (Simplified: Just taking distance to the target point)
        # For better accuracy in presentation, we project vector onto path normal
        dx_f = front_axle_x - path_x[target_idx]
        dy_f = front_axle_y - path_y[target_idx]
        
        # Determine sign of error (left or right of path)
        # We use the cross product of path vector and vehicle vector
        cross_track_error = np.hypot(dx_f, dy_f)
        perp_vec = [np.sin(path_yaw), -np.cos(path_yaw)]
        dot_prod = dx_f * perp_vec[0] + dy_f * perp_vec[1]
        
        if dot_prod > 0:
            cross_track_error = -cross_track_error

        # 3. Stanley Control Law
        # delta = heading_error + arctan( k * e / (v + k_soft) )
        cross_track_steering = np.arctan2(self.k * cross_track_error, (abs(v) + self.k_soft))
        
        steer_cmd = heading_error + cross_track_steering
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