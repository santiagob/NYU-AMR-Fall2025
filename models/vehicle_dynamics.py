import numpy as np

class VehicleParams:
    # Vehicle Parameters (Dynamics)
    WHEELBASE = 2.5  
    TRACK_WIDTH = 1.5  
    MAX_STEER = np.radians(60)  
    MASS = 1500.0  
    YAW_INERTIA = 2500.0   
    CG_TO_FRONT = 1.2      
    CG_TO_REAR = WHEELBASE - CG_TO_FRONT  
    
    # Tire Parameters (Linear Model)
    # Aggregate (both tires per axle) cornering stiffness; negative sign handled in dynamics
    C_ALPHA_FRONT = 25000.0  
    C_ALPHA_REAR = 25000.0   
    MU = 0.9  # road friction coefficient used for lateral saturation

    # Aerodynamics & Friction
    AIR_DENSITY = 1.2      
    DRAG_COEFF = 0.3       
    FRONTAL_AREA = 2.2     
    ROLL_RESIST = 0.01  
    MAX_ENGINE_FORCE = 8000.0  
    BRAKE_FORCE_COEFF = 0.8    
    LAT_AERO_DAMP = 40.0  # simple lateral damping term (N·s/m)

class DynamicBicycleModel:
    def __init__(self, x=0, y=0, yaw=0, vx=0, vy=0, r=0):
        self.state = np.array([x, y, yaw, vx, vy, r], dtype=float)
        self.params = VehicleParams()

    def update(self, throttle, steer_rad, dt):
        """ Runge-Kutta 2 (Heun's Method) Integration """
        k1 = self._compute_derivatives(self.state, throttle, steer_rad)
        state_pred = self.state + k1 * dt
        
        k2 = self._compute_derivatives(state_pred, throttle, steer_rad)
        self.state = self.state + 0.5 * (k1 + k2) * dt
        
        # Normalize Yaw to -pi to pi
        self.state[2] = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))

    def _compute_derivatives(self, state, throttle, steer):
        x, y, yaw, vx, vy, r = state
        p = self.params
        speed = max(np.hypot(vx, vy), 1e-4)
        EPS = 1e-4
        steer = np.clip(steer, -p.MAX_STEER, p.MAX_STEER)
        vx_safe = np.sign(vx) * max(abs(vx), EPS)

        # 1. Slip Angle Calculation (Kinematics -> Dynamics)
        alpha_f = np.arctan2(vy + p.CG_TO_FRONT * r, vx_safe) - steer
        alpha_r = np.arctan2(vy - p.CG_TO_REAR * r, vx_safe)

        # 2. Tire Forces (Linear Model)
        Fyf = -p.C_ALPHA_FRONT * alpha_f
        Fyr = -p.C_ALPHA_REAR * alpha_r
        
        # Simple lateral damping to bleed side slip at speed
        Fy_damp = -p.LAT_AERO_DAMP * vy * max(speed, EPS)
        Fyf += 0.5 * Fy_damp
        Fyr += 0.5 * Fy_damp

        # Lateral force saturation (per axle) using friction limit
        g = 9.81
        Fz_front = p.MASS * g * p.CG_TO_REAR / p.WHEELBASE
        Fz_rear = p.MASS * g * p.CG_TO_FRONT / p.WHEELBASE
        Fy_front_limit = p.MU * Fz_front
        Fy_rear_limit = p.MU * Fz_rear
        Fyf = np.clip(Fyf, -Fy_front_limit, Fy_front_limit)
        Fyr = np.clip(Fyr, -Fy_rear_limit, Fy_rear_limit)

        # 3. Longitudinal Forces
        if throttle >= 0:
            Fx = throttle * p.MAX_ENGINE_FORCE
        else:
            Fx = throttle * p.BRAKE_FORCE_COEFF * p.MASS * 9.81

        # Resistances (always oppose motion)
        Fdrag = 0.5 * p.AIR_DENSITY * p.DRAG_COEFF * p.FRONTAL_AREA * speed**2
        Froll = p.ROLL_RESIST * p.MASS * 9.81
        # Drag opposes velocity direction
        drag_dir = -vx_safe / (speed + EPS)  # Normalized direction opposite to motion
        Fx_net = Fx + drag_dir * (Fdrag + Froll)

        # 4. Equations of Motion (Newton-Euler)
        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dyaw = r
        dvx = (Fx_net + p.MASS * vy * r) / p.MASS
        dvy = (Fyf * np.cos(steer) + Fyr - p.MASS * vx * r) / p.MASS
        dr  = (p.CG_TO_FRONT * Fyf * np.cos(steer) - p.CG_TO_REAR * Fyr) / p.YAW_INERTIA

        return np.array([dx, dy, dyaw, dvx, dvy, dr])

    # Properties for unified interface with kinematic model
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