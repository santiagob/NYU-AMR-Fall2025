import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import csv

# --- Vehicle Parameters (Dynamics) ---
WHEELBASE = 2.5  # Distance between front and rear axles [m]
TRACK_WIDTH = 1.5  # Distance between left and right wheels [m]
MAX_STEER_ANGLE_DEG = 60  # Maximum steering angle [deg]

VEHICLE_MASS = 1500.0  # Vehicle mass [kg]
YAW_INERTIA = 2500.0   # Yaw moment of inertia [kg*m^2]
CG_TO_FRONT = 1.2      # Distance from CG to front axle [m]
CG_TO_REAR = WHEELBASE - CG_TO_FRONT  # Distance from CG to rear axle [m]

# --- Quasi-Kinematic Parameters (Minimal Lateral Motion) ---
# This configuration heavily constrains lateral dynamics to behave almost like kinematic model
# Result: Very minimal lateral sliding/drifting, car-like behavior

CORNERING_STIFFNESS_FRONT = 8000.0   # Front tire cornering stiffness [N/rad]
CORNERING_STIFFNESS_REAR = 8000.0    # Rear tire cornering stiffness [N/rad]

# Very high lateral damping to suppress lateral velocity
LATERAL_DAMPING_COEFF = 5000.0  # Damping force per unit lateral velocity [N·s/m]

# Kinematic constraint factor: forces lateral velocity toward kinematic prediction
# 0.0 = pure dynamic model, 1.0 = pure kinematic model
# 0.95 = heavily constrained (quasi-kinematic, minimal slip)
KINEMATIC_CONSTRAINT_FACTOR = 0.99  # How much to constrain toward kinematic behavior

# Tire slip angle saturation (simplified)
MAX_SLIP_ANGLE = np.radians(15)  # Maximum slip angle before severe grip loss

AIR_DENSITY = 1.2      # Air density [kg/m^3]
DRAG_COEFF = 0.3       # Aerodynamic drag coefficient
FRONTAL_AREA = 2.2     # Frontal area [m^2]
ROLLING_RESIST_COEFF = 0.015  # Increased for more realistic rolling resistance
MAX_ENGINE_FORCE = 8000.0  # Maximum engine force [N]
BRAKE_FORCE_COEFF = 0.8    # Braking force coefficient

# --- Simulation Parameters ---
SIM_DURATION = 15  # seconds
DT = 0.1  # seconds (time step)
INITIAL_STATE = [0, 0, np.pi/2, 0, 0, 0]  # [x, y, yaw, vx, vy, yaw_rate]

# --- External schedule logic  ---
# Allows loading a driving schedule from CSV file.
external_schedule = []
external_enabled = False

def set_external_schedule(schedule):
    global external_schedule, external_enabled
    external_schedule = list(schedule)
    external_enabled = bool(external_schedule)

def clear_external_schedule():
    global external_schedule, external_enabled
    external_schedule = []
    external_enabled = False

def load_schedule_from_csv(path):
    raw_rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
    expanded = []
    for row in raw_rows:
        try:
            start = float(row.get('start', 0.0))
            end = float(row.get('end', start))
        except Exception:
            continue
        mode = (row.get('mode') or '').strip().lower()
        steer = float(row.get('steer', 0.0)) if row.get('steer') not in (None, '') else 0.0
        if mode in ('ramp', 'accel'):
            v0 = float(row.get('v0', row.get('speed', 0.0)))
            if row.get('accel') not in (None, ''):
                accel = float(row.get('accel', 0.0))
            elif row.get('v1') not in (None, ''):
                v1 = float(row.get('v1'))
                duration = max(1e-6, end - start)
                accel = (v1 - v0) / duration
            else:
                accel = 0.0
            span = max(0.0, end - start)
            if span <= 0:
                continue
            n_steps = int(math.ceil(span / DT))
            for i in range(n_steps):
                t0 = start + i * DT
                t1 = min(end, t0 + DT)
                t0 = round(t0, 10)
                t1 = round(t1, 10)
                speed = v0 + accel * (t0 - start)
                expanded.append({'start': t0, 'end': t1, 'speed': speed, 'steer': steer})
        else:
            speed = float(row.get('speed', 0.0)) if row.get('speed') not in (None, '') else 0.0
            expanded.append({'start': start, 'end': end, 'speed': speed, 'steer': steer})
    set_external_schedule(expanded)

# --- Simplified Dynamic Bicycle Model ---
def compute_tire_force(slip_angle, cornering_stiffness):
    """
    Computes tire lateral force using a simple linear model with soft saturation.
    This simplified version reduces excessive lateral dynamics.
    
    Args:
        slip_angle: Tire slip angle [rad]
        cornering_stiffness: Linear cornering stiffness [N/rad]
    
    Returns:
        Lateral tire force [N]
    """
    # Simple linear model with gentle saturation
    # Clamp slip angle to reasonable range
    slip_clamped = np.clip(slip_angle, -MAX_SLIP_ANGLE, MAX_SLIP_ANGLE)
    
    # Linear tire force
    Fy = cornering_stiffness * slip_clamped
    
    return Fy

def compute_dynamics(state, throttle, steer_rad):
    """
    Computes the time derivatives of the state using an improved dynamic bicycle model.
    
    Key improvements:
    - Realistic positive cornering stiffness values
    - Slip angle saturation (tire saturation)
    - Improved lateral dynamics
    - Better numerical stability
    
    Args:
        state: Tuple (x, y, yaw, vx, vy, r)
        throttle: Throttle/brake command [-1, 1]
        steer_rad: Steering angle [rad]
    
    Returns:
        Tuple of state derivatives (dx, dy, dyaw, dvx, dvy, dr)
    """
    x, y, yaw, vx, vy, r = state
    speed = np.hypot(vx, vy)
    EPS = 1e-4

    # Handle very low speeds to avoid division by zero
    if abs(vx) < EPS:
        # Vehicle is essentially stopped
        return 0, 0, 0, 0, 0, 0
    
    # Clamp steering to maximum angle
    steer_rad = np.clip(steer_rad, -np.radians(MAX_STEER_ANGLE_DEG), 
                        np.radians(MAX_STEER_ANGLE_DEG))
    
    # KINEMATIC CONSTRAINT: Rear tires have NO lateral slip
    # In kinematic model: rear axle velocity is purely longitudinal (no lateral component)
    # This means: vy at rear axle = 0, so vy - CG_TO_REAR * r = 0
    # Therefore: vy_kinematic = CG_TO_REAR * r
    
    # Enforce kinematic relationship for lateral velocity
    vy_kinematic = CG_TO_REAR * r
    
    # Compute slip angles
    # Front slip angle: measured relative to front wheel orientation
    alpha_f = np.arctan2(vy + CG_TO_FRONT * r, vx) - steer_rad
    
    # KINEMATIC MODEL: Rear tire has ZERO slip angle (no lateral slip)
    # We eliminate rear tire lateral force entirely
    alpha_r = 0.0  # Kinematic constraint: no slip at rear
    
    # Compute tire forces
    Fyf = compute_tire_force(alpha_f, CORNERING_STIFFNESS_FRONT)
    Fyr = 0.0  # KINEMATIC: No lateral force at rear (no slip condition)
    
    # Longitudinal force (engine/brake)
    if vx >= 0:
        Fx = throttle * MAX_ENGINE_FORCE if throttle >= 0 else throttle * BRAKE_FORCE_COEFF * VEHICLE_MASS * 9.81
    else:
        Fx = throttle * MAX_ENGINE_FORCE if throttle <= 0 else throttle * BRAKE_FORCE_COEFF * VEHICLE_MASS * 9.81

    # Drag and rolling resistance
    Fdrag = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA * speed**2
    Froll = ROLLING_RESIST_COEFF * VEHICLE_MASS * 9.81
    Fx_net = Fx - Fdrag - Froll

    # Equations of motion (bicycle model)
    dx = vx * np.cos(yaw) - vy * np.sin(yaw)
    dy = vx * np.sin(yaw) + vy * np.cos(yaw)
    dyaw = r

    # KINEMATIC CONSTRAINT: Force lateral velocity to satisfy no-slip condition at rear axle
    # Kinematic model: rear axle has no lateral velocity component
    # This means: vy_rear = vy - CG_TO_REAR * r = 0
    # Therefore: vy must equal CG_TO_REAR * r
    
    # Apply very strong constraint to force vy toward kinematic value
    Fy_damping = -LATERAL_DAMPING_COEFF * vy  # Damping
    Fy_constraint = -KINEMATIC_CONSTRAINT_FACTOR * VEHICLE_MASS * 10.0 * (vy - vy_kinematic)  # Strong kinematic constraint
    
    # Lateral dynamics (Fyr = 0 because of kinematic constraint)
    dvy = (Fyf * np.cos(steer_rad) + Fy_damping + Fy_constraint) / VEHICLE_MASS - vx * r
    
    # Longitudinal acceleration
    dvx = Fx_net / VEHICLE_MASS + vy * r
    
    # Yaw acceleration: moments from front and rear tire forces
    dr = (CG_TO_FRONT * Fyf * np.cos(steer_rad) - CG_TO_REAR * Fyr) / YAW_INERTIA
    
    return dx, dy, dyaw, dvx, dvy, dr

def propagate_state_dynamic(state, throttle, steer_deg, dt):
    """
    Integrates state forward with KINEMATIC constraint: rear tire has NO lateral slip.
    The kinematic model dictates: vy_rear = vy - CG_TO_REAR * r = 0
    Therefore: vy = CG_TO_REAR * r (strictly enforced)
    """
    steer_rad = np.radians(np.clip(steer_deg, -MAX_STEER_ANGLE_DEG, MAX_STEER_ANGLE_DEG))
    
    # Heun's method (improved Euler)
    k1 = compute_dynamics(state, throttle, steer_rad)
    state_pred = tuple(s + k * dt for s, k in zip(state, k1))
    k2 = compute_dynamics(state_pred, throttle, steer_rad)
    new_state = tuple(s + 0.5 * (k1i + k2i) * dt for s, k1i, k2i in zip(state, k1, k2))
    
    # CRITICAL: Apply kinematic constraint to eliminate rear tire lateral slip
    # Kinematic model: rear axle has zero lateral velocity (no-slip condition)
    # This forces: vy = CG_TO_REAR * r (exactly)
    new_state = list(new_state)
    x, y, yaw, vx, vy, r = new_state
    
    # Reconstruct vy from kinematic no-slip condition
    # This ensures rear tire has ZERO lateral velocity
    vy_kinematic = CG_TO_REAR * r
    new_state[4] = vy_kinematic  # Set vy to kinematic value
    
    # Normalize yaw
    new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))
    
    return tuple(new_state)

# --- Simple speed controller for schedule compatibility ---
def speed_to_throttle(target_speed, current_vx):
    """
    Simple proportional controller to convert speed error to throttle command.
    """
    # Proportional controller
    Kp = 0.5
    error = target_speed - current_vx
    throttle = Kp * error
    throttle = np.clip(throttle, -1.0, 1.0)
    return throttle

# --- Vehicle class for visualization (unchanged except state) ---
class Vehicle:
    """
    Represents the vehicle and provides methods for state update and visualization.
    """
    def __init__(self, x, y, yaw, vx, vy, r, wheelbase, track_width, max_steer_angle_rad, wheel_length=0.6, wheel_width=0.5):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.r = r
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.max_steer_angle_rad = max_steer_angle_rad
        self.body_length = wheelbase + 2* wheel_length
        self.body_width = track_width + wheel_width
        self.rear_overhang = wheel_length
        self.front_overhang = self.body_length - self.wheelbase - self.rear_overhang
        self.wheel_length = wheel_length
        self.wheel_width = wheel_width

    def update(self, dt, target_speed, target_steer_angle_deg):
        """
        Updates the vehicle state given speed and steering commands.
        """
        throttle = speed_to_throttle(target_speed, self.vx)
        new_state = propagate_state_dynamic(
            (self.x, self.y, self.yaw, self.vx, self.vy, self.r),
            throttle, target_steer_angle_deg, dt
        )
        self.x, self.y, self.yaw, self.vx, self.vy, self.r = new_state
        return np.radians(np.clip(target_steer_angle_deg, -MAX_STEER_ANGLE_DEG, MAX_STEER_ANGLE_DEG))

    def get_car_body_vertices(self):
        """
        Returns the coordinates of the car body corners for plotting.
        """
        half_width = self.body_width / 2
        vertices_local = np.array([
            [-self.rear_overhang, half_width],
            [self.wheelbase + self.front_overhang, half_width],
            [self.wheelbase + self.front_overhang, -half_width],
            [-self.rear_overhang, -half_width],
            [-self.rear_overhang, half_width]
        ])
        rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)]
        ])
        rotated_vertices = (rotation_matrix @ vertices_local.T).T
        global_vertices_x = self.x + rotated_vertices[:, 0]
        global_vertices_y = self.y + rotated_vertices[:, 1]
        return global_vertices_x, global_vertices_y

    def get_wheel_vertices(self, steer_angle_rad):
        """
        Returns the coordinates of the wheels for plotting.
        """
        wheel_half_len = self.wheel_length / 2
        wheel_half_width = self.wheel_width / 2
        half_track = self.track_width / 2
        wheel_centers_local = {
            'RL': np.array([0, half_track]),
            'RR': np.array([0, -half_track]),
            'FL': np.array([self.wheelbase, half_track]),
            'FR': np.array([self.wheelbase, -half_track]),
        }
        delta_fl = delta_fr = 0
        if abs(steer_angle_rad) > 1e-6:
            turn_radius_rear_axle = self.wheelbase / np.tan(steer_angle_rad)
            delta_fl = np.arctan(self.wheelbase / (turn_radius_rear_axle - np.sign(steer_angle_rad) * half_track))
            delta_fr = np.arctan(self.wheelbase / (turn_radius_rear_axle + np.sign(steer_angle_rad) * half_track))
        wheel_angles = {'RL': 0, 'RR': 0, 'FL': delta_fl, 'FR': delta_fr}
        all_wheel_vertices = []
        wheel_vertices_local_template = np.array([
            [-wheel_half_len, wheel_half_width],
            [wheel_half_len, wheel_half_width],
            [wheel_half_len, -wheel_half_width],
            [-wheel_half_len, -wheel_half_width],
            [-wheel_half_len, wheel_half_width]
        ])
        vehicle_rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)]
        ])
        for wheel_key in ['FL', 'FR', 'RL', 'RR']:
            center_local = wheel_centers_local[wheel_key]
            wheel_angle = wheel_angles[wheel_key]
            wheel_rotation_matrix = np.array([
                [np.cos(wheel_angle), -np.sin(wheel_angle)],
                [np.sin(wheel_angle), np.cos(wheel_angle)]
            ])
            rotated_wheel_vertices = (wheel_rotation_matrix @ wheel_vertices_local_template.T).T
            wheel_vertices_in_vehicle_frame = rotated_wheel_vertices + center_local
            global_wheel_vertices = (vehicle_rotation_matrix @ wheel_vertices_in_vehicle_frame.T).T + np.array([self.x, self.y])
            all_wheel_vertices.append((global_wheel_vertices[:, 0], global_wheel_vertices[:, 1]))
        return all_wheel_vertices

# --- Batch simulation for export ---
def batch_simulate(schedule=None, duration=None):
    """
    Runs a batch simulation using a schedule and returns the state histories.
    """
    vehicle = Vehicle(
        x=INITIAL_STATE[0], y=INITIAL_STATE[1], yaw=INITIAL_STATE[2],
        vx=INITIAL_STATE[3], vy=INITIAL_STATE[4], r=INITIAL_STATE[5],
        wheelbase=WHEELBASE, track_width=TRACK_WIDTH,
        max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
    )
    sched = schedule if schedule is not None else external_schedule
    if duration is None:
        if sched:
            duration = max((entry.get('end', 0.0) for entry in sched), default=SIM_DURATION)
        else:
            duration = SIM_DURATION
    num_steps = int(math.ceil(duration / DT))
    times, xs, ys, yaws, vxs, vys, rs, speeds, steers, yaw_rates, lat_accels = [], [], [], [], [], [], [], [], [], [], []
    for i in range(num_steps):
        t = i * DT
        scheduled = None
        if sched:
            for entry in sched:
                if entry['start'] <= t < entry['end']:
                    scheduled = entry
                    break
        if scheduled is not None:
            target_speed = scheduled.get('speed', 0.0)
            target_steer = scheduled.get('steer', 0.0)
        else:
            target_speed = 0.0
            target_steer = 0.0
        used_steer = vehicle.update(DT, target_speed, target_steer)
        times.append(t)
        xs.append(vehicle.x)
        ys.append(vehicle.y)
        yaws.append(vehicle.yaw)
        vxs.append(vehicle.vx)
        vys.append(vehicle.vy)
        rs.append(vehicle.r)
        speeds.append(np.hypot(vehicle.vx, vehicle.vy))
        steers.append(np.degrees(used_steer))
        yaw_rates.append(vehicle.r)
        lat_accels.append(vehicle.vx * vehicle.r)
    return {
        't': np.array(times),
        'x': np.array(xs),
        'y': np.array(ys),
        'yaw': np.array(yaws),
        'vx': np.array(vxs),
        'vy': np.array(vys),
        'r': np.array(rs),
        'speed': np.array(speeds),
        'steer': np.array(steers),
        'yaw_rate': np.array(yaw_rates),
        'lat_accel': np.array(lat_accels),
    }

def export_simulation_images(path, histories, show_car=False):
    """
    Exports trajectory and dynamics plots to image files.
    """
    base, _ = os.path.splitext(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig1, ax = plt.subplots(figsize=(8, 8))
    ax.plot(histories['x'], histories['y'], '-b')
    ax.scatter(histories['x'][0], histories['y'][0], c='g', label='start')
    ax.scatter(histories['x'][-1], histories['y'][-1], c='r', label='end')
    try:
        yaw0 = float(histories['yaw'][0])
        yawf = float(histories['yaw'][-1])
        arrow_len = max(0.5, np.linalg.norm([histories['x'][-1]-histories['x'][0], histories['y'][-1]-histories['y'][0]]) * 0.05)
        ax.arrow(histories['x'][0], histories['y'][0], arrow_len * math.cos(yaw0), arrow_len * math.sin(yaw0), head_width=0.2, head_length=0.3, fc='g', ec='g')
        ax.arrow(histories['x'][-1], histories['y'][-1], arrow_len * math.cos(yawf), arrow_len * math.sin(yawf), head_width=0.2, head_length=0.3, fc='r', ec='r')
    except Exception:
        pass
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Aerial Trajectory')
    ax.legend()
    fig1.tight_layout()
    traj_path = f"{base}_trajectory.png"
    fig1.savefig(traj_path, dpi=200)
    plt.close(fig1)
    fig2, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    t = histories['t']
    axes[0].plot(t, histories['speed'], '-k')
    axes[0].set_title('Speed [m/s]')
    axes[1].plot(t, histories['steer'], '-r')
    axes[1].set_title('Steer [deg]')
    axes[2].plot(t, histories['yaw_rate'], '-m')
    axes[2].set_title('Yaw Rate [rad/s]')
    axes[3].plot(t, histories['lat_accel'], '-c')
    axes[3].set_title('Lateral Accel [m/s^2]')
    yaw_deg = np.degrees(histories['yaw'])
    axes[4].plot(t, yaw_deg, '-g')
    axes[4].set_title('Yaw [deg]')
    for axx in axes:
        axx.set_xlabel('Time [s]')
    fig2.tight_layout()
    dyn_path = f"{base}_dynamics.png"
    fig2.savefig(dyn_path, dpi=200)
    plt.close(fig2)
    return traj_path, dyn_path

def save_histories_csv(path, histories):
    """
    Saves simulation histories to a CSV file.
    """
    base, _ = os.path.splitext(path)
    csv_path = f"{base}_history.csv"
    keys = ['t', 'x', 'y', 'yaw', 'vx', 'vy', 'r', 'speed', 'steer', 'yaw_rate', 'lat_accel']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(histories['t'])
        for i in range(n):
            w.writerow([histories[k][i] for k in keys])
    return csv_path

# --- Main Simulation Loop (GUI) ---
def main():
    """
    Runs the interactive simulation with keyboard controls and visualization.
    """
    vehicle = Vehicle(
        x=INITIAL_STATE[0], y=INITIAL_STATE[1], yaw=INITIAL_STATE[2],
        vx=INITIAL_STATE[3], vy=INITIAL_STATE[4], r=INITIAL_STATE[5],
        wheelbase=WHEELBASE, track_width=TRACK_WIDTH,
        max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
    )
    x_history = [vehicle.x]
    y_history = [vehicle.y]
    yaw_history = [vehicle.yaw]
    vx_history = [vehicle.vx]
    vy_history = [vehicle.vy]
    r_history = [vehicle.r]
    speed_history = [np.hypot(vehicle.vx, vehicle.vy)]
    steer_angle_history_rad = [0.0]
    current_speed_cmd = 0.0
    current_steer_cmd = 0.0
    paused = False
    held_keys = set()
    SPEED_INC_PER_SEC = 1.0
    STEER_INC_PER_SEC = 40.0

    def reset_sim():
        """
        Resets the simulation to the initial state.
        """
        nonlocal vehicle, x_history, y_history, yaw_history, vx_history, vy_history, r_history, speed_history, steer_angle_history_rad, current_speed_cmd, current_steer_cmd
        vehicle.x, vehicle.y, vehicle.yaw, vehicle.vx, vehicle.vy, vehicle.r = INITIAL_STATE
        current_speed_cmd = 0.0
        current_steer_cmd = 0.0
        x_history = [vehicle.x]
        y_history = [vehicle.y]
        yaw_history = [vehicle.yaw]
        vx_history = [vehicle.vx]
        vy_history = [vehicle.vy]
        r_history = [vehicle.r]
        speed_history = [np.hypot(vehicle.vx, vehicle.vy)]
        steer_angle_history_rad = [0.0]

    def on_key_press(event):
        """
        Handles keyboard press events for user controls.
        """
        nonlocal paused, current_speed_cmd
        key = event.key
        if key is None:
            return
        if key in ('left', 'right', 'up', 'down'):
            held_keys.add(key)
        elif key == ' ' or key == 'space':
            paused = not paused
        elif key == 'r':
            reset_sim()
        elif key == 'e':
            global external_enabled
            external_enabled = not external_enabled
        elif key == 's':
            held_keys.discard('up')
            held_keys.discard('down')
            current_speed_cmd = 0.0
            vehicle.vx = 0.0
            vehicle.vy = 0.0
        # ignore other keys

    def on_key_release(event):
        """
        Handles keyboard release events for user controls.
        """
        key = event.key
        if key in held_keys:
            held_keys.discard(key)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')
    path_line, = ax.plot([], [], 'b-', linewidth=1.5, label='Vehicle Path')
    car_body_patch, = ax.plot([], [], 'k-', linewidth=3, label='Vehicle Body')
    wheel_patches = [ax.plot([], [], 'k-', linewidth=2)[0] for _ in range(4)]
    current_title = ax.set_title('')
    schedule_status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    def init():
        """
        Initializes the animation.
        """
        path_line.set_data([], [])
        car_body_patch.set_data([], [])
        for wheel_patch in wheel_patches:
            wheel_patch.set_data([], [])
        current_title.set_text('')
        return [path_line, car_body_patch, *wheel_patches, current_title]

    def update(frame):
        """
        Updates the animation for each frame.
        """
        nonlocal current_speed_cmd, current_steer_cmd, paused
        if not paused:
            if 'up' in held_keys:
                current_speed_cmd += SPEED_INC_PER_SEC * DT
            if 'down' in held_keys:
                current_speed_cmd -= SPEED_INC_PER_SEC * DT
            if 'left' in held_keys:
                current_steer_cmd += STEER_INC_PER_SEC * DT
            if 'right' in held_keys:
                current_steer_cmd -= STEER_INC_PER_SEC * DT
            if external_enabled and external_schedule:
                t = (len(x_history) - 1) * DT
                matched = None
                for entry in external_schedule:
                    if entry['start'] <= t < entry['end']:
                        matched = entry
                        break
                if matched is not None:
                    scheduled_speed = matched.get('speed', current_speed_cmd)
                    scheduled_steer = matched.get('steer', current_steer_cmd)
                    current_speed_cmd = scheduled_speed
                    current_steer_cmd = scheduled_steer
            current_steer_cmd = max(-MAX_STEER_ANGLE_DEG, min(MAX_STEER_ANGLE_DEG, current_steer_cmd))
            used_steer = vehicle.update(DT, current_speed_cmd, current_steer_cmd)
            x_history.append(vehicle.x)
            y_history.append(vehicle.y)
            yaw_history.append(vehicle.yaw)
            vx_history.append(vehicle.vx)
            vy_history.append(vehicle.vy)
            r_history.append(vehicle.r)
            speed_history.append(np.hypot(vehicle.vx, vehicle.vy))
            steer_angle_history_rad.append(used_steer)
        idx = -1
        path_line.set_data(x_history, y_history)
        temp_vehicle = Vehicle(
            x=x_history[idx], y=y_history[idx], yaw=yaw_history[idx],
            vx=vx_history[idx], vy=vy_history[idx], r=r_history[idx],
            wheelbase=WHEELBASE, track_width=TRACK_WIDTH,
            max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
        )
        car_x, car_y = temp_vehicle.get_car_body_vertices()
        car_body_patch.set_data(car_x, car_y)
        wheel_vertices = temp_vehicle.get_wheel_vertices(steer_angle_history_rad[idx])
        for i, (wx, wy) in enumerate(wheel_vertices):
            wheel_patches[i].set_data(wx, wy)
        current_time = (len(x_history) - 1) * DT
        current_speed = speed_history[idx]
        current_title.set_text(f'Ackermann Dynamic Vehicle Simulation (Time: {current_time:.1f} s, Speed: {current_speed:.2f} m/s)')
        if external_enabled and external_schedule:
            active = None
            t = (len(x_history) - 1) * DT
            for entry in external_schedule:
                if entry['start'] <= t < entry['end']:
                    active = entry
                    break
            if active is not None:
                schedule_status_text.set_text(f'Schedule: ON  | [{active["start"]:.1f},{active["end"]:.1f}]s speed={active["speed"]}m/s steer={active["steer"]}deg')
            else:
                schedule_status_text.set_text('Schedule: ON  | no active entry')
        else:
            schedule_status_text.set_text('Schedule: OFF')
        margin = 10
        ax.set_xlim(x_history[idx] - margin, x_history[idx] + margin)
        ax.set_ylim(y_history[idx] - margin, y_history[idx] + margin)
        return [path_line, car_body_patch, *wheel_patches, current_title]

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    ani = animation.FuncAnimation(
        fig, update, frames=range(1000000),
        init_func=init, blit=False, interval=DT * 1000, repeat=False
    )
    plt.show()

if __name__ == "__main__":
    """
    Entry point for running the simulator as a script.
    Supports command-line arguments for schedule loading and export.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Ackermann dynamic simulator (interactive)')
    parser.add_argument('--schedule', '-s', help='Path to schedule CSV to load at startup')
    parser.add_argument('--enable', action='store_true', help='Enable loaded schedule immediately')
    parser.add_argument('--export', '-x', help='Export simulation image to PATH (runs headless)')
    parser.add_argument('--duration', type=float, help='Duration (s) to use for export run (overrides schedule end)')
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schedules_dir = os.path.join(script_dir, 'schedules')
    default_output_dir = os.path.join(script_dir, 'output')
    schedule_path = None
    tried_paths = []
    if args.schedule:
        tried_paths.append(os.path.abspath(args.schedule))
        if os.path.exists(args.schedule):
            schedule_path = args.schedule
        else:
            candidate = os.path.join(schedules_dir, args.schedule)
            tried_paths.append(os.path.abspath(candidate))
            if os.path.exists(candidate):
                schedule_path = candidate
            else:
                candidate2 = os.path.join(schedules_dir, args.schedule + '.csv')
                tried_paths.append(os.path.abspath(candidate2))
                if os.path.exists(candidate2):
                    schedule_path = candidate2
    if args.schedule and not schedule_path:
        import sys
        RED = '\033[91m'
        RESET = '\033[0m'
        print(f"{RED}ERROR: schedule file not found. Tried the following absolute paths:{RESET}")
        for p in tried_paths:
            print('  -', p)
        sys.exit(1)
    if schedule_path:
        try:
            load_schedule_from_csv(schedule_path)
            print(f'Loaded schedule from {schedule_path}')
        except Exception as e:
            print(f'Failed to load schedule from {schedule_path}: {e}')
    if args.enable:
        external_enabled = True
    if args.export:
        os.makedirs(default_output_dir, exist_ok=True)
        histories = batch_simulate(schedule=external_schedule if external_schedule else None, duration=args.duration)
        export_arg = args.export
        if os.path.sep not in export_arg:
            export_path = os.path.join(default_output_dir, export_arg)
        else:
            parts = export_arg.split(os.path.sep)
            if parts[0] == 'output':
                export_path = os.path.join(default_output_dir, *parts[1:])
            else:
                export_path = export_arg
        export_parent = os.path.dirname(os.path.abspath(export_path))
        if export_parent:
            os.makedirs(export_parent, exist_ok=True)
        traj_path, dyn_path = export_simulation_images(export_path, histories)
        print(f'Exported trajectory to {traj_path}')
        print(f'Exported dynamics to {dyn_path}')
        csv_path = save_histories_csv(export_path, histories)
        print(f'Exported histories CSV to {csv_path}')
    else:
        main()
