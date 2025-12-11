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
    # Using your values: -600 N/rad seems low for a real car (usually ~1000+), 
    # but we will keep your tuning to ensure stability with your previous testing.
    C_ALPHA_FRONT = -600.0  
    C_ALPHA_REAR = -600.0   

    # Aerodynamics & Friction
    AIR_DENSITY = 1.2      
    DRAG_COEFF = 0.3       
    FRONTAL_AREA = 2.2     
    ROLL_RESIST = 0.01  
    MAX_ENGINE_FORCE = 8000.0  
    BRAKE_FORCE_COEFF = 0.8    

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
        speed = np.hypot(vx, vy)
        EPS = 1e-4

        # 1. Slip Angle Calculation (Kinematics -> Dynamics)
        if abs(vx) < EPS:
            alpha_f, alpha_r = 0.0, 0.0
        else:
            alpha_f = (vy + p.CG_TO_FRONT * r) / abs(vx) - steer
            alpha_r = (vy - p.CG_TO_REAR * r) / abs(vx)

        # 2. Tire Forces (Linear Model)
        Fyf = p.C_ALPHA_FRONT * alpha_f
        Fyr = p.C_ALPHA_REAR * alpha_r
        
        # Handle reverse logic
        if vx < 0:
            Fyf, Fyr = -Fyf, -Fyr

        # 3. Longitudinal Forces
        if throttle >= 0:
            Fx = throttle * p.MAX_ENGINE_FORCE
        else:
            Fx = throttle * p.BRAKE_FORCE_COEFF * p.MASS * 9.81

        # Resistances
        Fdrag = 0.5 * p.AIR_DENSITY * p.DRAG_COEFF * p.FRONTAL_AREA * speed**2
        Froll = p.ROLL_RESIST * p.MASS * 9.81
        drag_dir = -1 if vx > 0 else 1
        Fx_net = Fx + (drag_dir * (Fdrag + Froll))

        # 4. Equations of Motion (Newton-Euler)
        dx = vx * np.cos(yaw) - vy * np.sin(yaw)
        dy = vx * np.sin(yaw) + vy * np.cos(yaw)
        dyaw = r
        dvx = (Fx_net + p.MASS * vy * r) / p.MASS
        dvy = (Fyf * np.cos(steer) + Fyr - p.MASS * vx * r) / p.MASS
        dr  = (p.CG_TO_FRONT * Fyf * np.cos(steer) - p.CG_TO_REAR * Fyr) / p.YAW_INERTIA

        return np.array([dx, dy, dyaw, dvx, dvy, dr])