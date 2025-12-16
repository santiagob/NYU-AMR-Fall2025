"""
Kinematic-Based Vehicle Model
==============================

A simplified bicycle model based purely on kinematics, without complex 
tire dynamics or lateral slip modeling.

This model enforces:
- Rear axle no-slip constraint: vy = CG_TO_REAR * r
- Yaw rate from steering: r = (vx/WHEELBASE) * tan(steer)
- No tire slip angles or complex forces

Advantages:
- Numerically stable and predictable
- Fast computation
- No drift or excessive lateral movement
- Easy to tune and understand

Use when:
- You need stable, realistic vehicle behavior
- Computational performance is important
- Simple steering response is desired
"""

import numpy as np


class VehicleKinematicParams:
    """Vehicle parameters for kinematic model (same as dynamic model for consistency)"""
    WHEELBASE = 2.5
    TRACK_WIDTH = 1.5
    MAX_STEER = np.radians(60)
    MASS = 1500.0
    YAW_INERTIA = 2500.0
    CG_TO_FRONT = 1.2
    CG_TO_REAR = WHEELBASE - CG_TO_FRONT

    # Simple speed control gains
    KP_SPEED = 0.5  # Proportional gain for speed control


class KinematicBicycleModel:
    """
    Pure kinematic bicycle model with rear-axle no-slip constraint.
    
    State: [x, y, yaw, vx, vy, r]
    - x, y: Position of rear axle center
    - yaw: Vehicle heading angle (radians)
    - vx: Longitudinal velocity (m/s)
    - vy: Lateral velocity (m/s)
    - r: Yaw rate (rad/s)
    
    The model enforces:
    - vy = CG_TO_REAR * r  (rear axle no-slip)
    - r = (vx/WHEELBASE) * tan(steer)  (bicycle kinematics)
    """

    def __init__(self, x=0, y=0, yaw=0, vx=0, vy=0, r=0):
        self.state = np.array([x, y, yaw, vx, vy, r], dtype=float)
        self.params = VehicleKinematicParams()
        self.last_steer_rad = 0.0  # Store steering for constraint enforcement

    def update(self, throttle, steer_rad, dt):
        """
        Update vehicle state using kinematic model with Heun's method (RK2).
        
        Args:
            throttle: Engine throttle [-1, 1]
            steer_rad: Steering angle in radians
            dt: Time step in seconds
        """
        # Store steering for constraint enforcement
        self.last_steer_rad = np.clip(steer_rad, -self.params.MAX_STEER, self.params.MAX_STEER)
        
        # Heun's method (RK2) for better accuracy
        k1 = self._compute_derivatives(self.state, throttle, self.last_steer_rad)
        state_pred = self.state + k1 * dt

        k2 = self._compute_derivatives(state_pred, throttle, self.last_steer_rad)
        self.state = self.state + 0.5 * (k1 + k2) * dt

        # Enforce kinematic constraints strictly
        self._enforce_kinematic_constraint()

        # Normalize yaw to [-pi, pi]
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

    def _compute_derivatives(self, state, throttle, steer):
        """Compute state derivatives from kinematic model"""
        x, y, yaw, vx, vy, r = state
        p = self.params

        # Longitudinal dynamics: simple acceleration model
        # Target speed from throttle
        target_vx = throttle * 10.0  # Max speed ~10 m/s when throttle=1
        dvx = p.KP_SPEED * (target_vx - vx)

        # Kinematics: bicycle model for yaw rate
        if abs(vx) > 1e-4:
            r_kinematic = (vx / p.WHEELBASE) * np.tan(np.clip(steer, -p.MAX_STEER, p.MAX_STEER))
        else:
            r_kinematic = 0.0

        # Lateral velocity from kinematic constraint (rear axle no-slip)
        vy_kinematic = p.CG_TO_REAR * r_kinematic

        # Position derivatives
        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dyaw = r_kinematic

        # Lateral velocity rate of change (approach kinematic value)
        dvy = (vy_kinematic - vy) / 0.1  # Quick convergence to kinematic value

        # Yaw rate derivative (approach kinematic value)
        dr = (r_kinematic - r) / 0.1

        return np.array([dx, dy, dyaw, dvx, dvy, dr])

    def _enforce_kinematic_constraint(self):
        """
        Strictly enforce the kinematic constraint after integration.
        
        This ensures:
        1. Rear axle has zero lateral velocity: vy = CG_TO_REAR * r
        2. Yaw rate is purely kinematic: r = (vx/WHEELBASE) * tan(steer)
        """
        x, y, yaw, vx, vy, r = self.state
        p = self.params

        # Compute kinematic yaw rate from current vx and last steering command
        if abs(vx) > 1e-4:
            steer_clipped = np.clip(self.last_steer_rad, -p.MAX_STEER, p.MAX_STEER)
            r_kinematic = (vx / p.WHEELBASE) * np.tan(steer_clipped)
        else:
            r_kinematic = 0.0

        # Enforce kinematic constraint: rear axle no-slip
        vy_kinematic = p.CG_TO_REAR * r_kinematic

        # Update state with kinematic values
        self.state[4] = vy_kinematic  # vy (rear axle lateral velocity = 0)
        self.state[5] = r_kinematic   # r (yaw rate from kinematics)

    def get_state(self):
        """Return current state [x, y, yaw, vx, vy, r]"""
        return self.state.copy()

    def set_state(self, state):
        """Set state directly"""
        self.state = np.array(state, dtype=float)

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def yaw(self):
        return self.state[2]

    @property
    def vx(self):
        return self.state[3]

    @property
    def vy(self):
        return self.state[4]

    @property
    def r(self):
        return self.state[5]
