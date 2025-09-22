import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- Vehicle Parameters ---
WHEELBASE = 2.5  # meters (L)
TRACK_WIDTH = 1.5  # meters (W)
MAX_STEER_ANGLE_DEG = 60  # degrees

# External control schedule (list of dicts with keys: start, end, speed, steer)
# Example: [{'start':0.0,'end':3.0,'speed':2.0,'steer':10.0}, ...]
external_schedule = []
external_enabled = False

def set_external_schedule(schedule):
    """Set the external schedule programmatically.

    `schedule` should be a list of dicts with numeric keys: `start`, `end`,
    `speed`, `steer` (degrees). Times are in seconds.
    """
    global external_schedule, external_enabled
    external_schedule = list(schedule)
    external_enabled = bool(external_schedule)

def clear_external_schedule():
    """Clear any external schedule and disable external control."""
    global external_schedule, external_enabled
    external_schedule = []
    external_enabled = False

def load_schedule_from_csv(path):
    """Load an external schedule from a CSV file.

    CSV columns (header optional): start,end,speed,steer
    Times in seconds, steer in degrees.
    """
    import csv
    raw_rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)

    # Expand parametric rows (ramp/accel/const) into per-DT entries
    expanded = []
    for row in raw_rows:
        # Parse common fields
        try:
            start = float(row.get('start', 0.0))
            end = float(row.get('end', start))
        except Exception:
            continue

        mode = (row.get('mode') or '').strip().lower()
        steer = float(row.get('steer', 0.0)) if row.get('steer') not in (None, '') else 0.0

        if mode in ('ramp', 'accel'):
            # parametric ramp: v0 and accel OR v0 and v1
            v0 = float(row.get('v0', row.get('speed', 0.0)))
            if row.get('accel') not in (None, ''):
                accel = float(row.get('accel', 0.0))
            elif row.get('v1') not in (None, ''):
                v1 = float(row.get('v1'))
                duration = max(1e-6, end - start)
                accel = (v1 - v0) / duration
            else:
                accel = 0.0

            # Number of DT steps
            span = max(0.0, end - start)
            if span <= 0:
                continue
            n_steps = int(math.ceil(span / DT))
            for i in range(n_steps):
                t0 = start + i * DT
                t1 = min(end, t0 + DT)
                # Round boundaries to avoid tiny floating-point gaps between
                # adjacent intervals (these gaps can cause a ``None`` match
                # when searching for the interval that contains a time t).
                t0 = round(t0, 10)
                t1 = round(t1, 10)
                speed = v0 + accel * (t0 - start)
                expanded.append({'start': t0, 'end': t1, 'speed': speed, 'steer': steer})
        else:
            # Treat as constant-speed entry (old format)
            speed = float(row.get('speed', 0.0)) if row.get('speed') not in (None, '') else 0.0
            # Keep as a single interval (the matching logic uses interval membership)
            expanded.append({'start': start, 'end': end, 'speed': speed, 'steer': steer})

    set_external_schedule(expanded)


def compute_kinematics(state, speed, steer_rad, wheelbase):
    """Compute instantaneous kinematic derivatives for the bicycle model.

    Args:
        state (tuple): (x, y, yaw) current state.
        speed (float): longitudinal speed (m/s).
        steer_rad (float): front wheel steer angle (radians).
        wheelbase (float): vehicle wheelbase (m).

    Returns:
        tuple: (dx, dy, dyaw) kinematic derivatives.
    """
    x, y, yaw = state

    # If steer is effectively zero, dyaw is zero and motion is straight
    if abs(steer_rad) < 1e-8:
        dx = speed * math.cos(yaw)
        dy = speed * math.sin(yaw)
        dyaw = 0.0
    else:
        # Bicycle model: yaw rate = v / L * tan(delta)
        dyaw = (speed / wheelbase) * math.tan(steer_rad)
        # Use the instantaneous yaw to compute x/y velocities
        dx = speed * math.cos(yaw)
        dy = speed * math.sin(yaw)

    return dx, dy, dyaw


def propagate_state(state, speed, steer_deg, dt, wheelbase):
    """Propagate the vehicle state for one time step using simple Euler integration.

    Args:
        state (tuple): (x, y, yaw) current state.
        speed (float): longitudinal speed (m/s).
        steer_deg (float): steering angle in degrees (will be clipped by
                           `MAX_STEER_ANGLE_DEG` before use).
        dt (float): time step (s).
        wheelbase (float): vehicle wheelbase (m).

    Returns:
        tuple: (x_new, y_new, yaw_new, used_steer_rad)
    """
    # Clip and convert steer angle
    steer_deg_clipped = max(-MAX_STEER_ANGLE_DEG, min(MAX_STEER_ANGLE_DEG, steer_deg))
    steer_rad = math.radians(steer_deg_clipped)

    # Compute kinematic derivatives
    dx, dy, dyaw = compute_kinematics(state, speed, steer_rad, wheelbase)

    # Simple Euler integration
    x, y, yaw = state
    x_new = x + dx * dt
    y_new = y + dy * dt
    yaw_new = yaw + dyaw * dt

    # Normalize yaw
    yaw_new = math.atan2(math.sin(yaw_new), math.cos(yaw_new))

    return x_new, y_new, yaw_new, steer_rad


def batch_simulate(schedule=None, duration=None):
    """Run a deterministic simulation from INITIAL_STATE using the provided
    external schedule (list of dicts). Returns a dict of histories.

    If `duration` is None and schedule is provided, use the max `end` time
    found in the schedule. Otherwise use `SIM_DURATION`.
    """
    # Initialize vehicle for batch run
    vehicle = Vehicle(
        x=INITIAL_STATE[0], y=INITIAL_STATE[1], yaw=INITIAL_STATE[2],
        speed=INITIAL_SPEED, wheelbase=WHEELBASE, track_width=TRACK_WIDTH,
        max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
    )

    sched = schedule if schedule is not None else external_schedule

    if duration is None:
        if sched:
            duration = max((entry.get('end', 0.0) for entry in sched), default=SIM_DURATION)
        else:
            duration = SIM_DURATION

    num_steps = int(math.ceil(duration / DT))

    times = []
    xs = []
    ys = []
    yaws = []
    speeds = []
    steers = []
    yaw_rates = []
    lat_accels = []

    state = (vehicle.x, vehicle.y, vehicle.yaw)

    for i in range(num_steps):
        t = i * DT
        # Find scheduled entry for this time
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

        # Propagate
        x_new, y_new, yaw_new, used_steer_rad = propagate_state(state, target_speed, target_steer, DT, WHEELBASE)

        # Compute derived dynamics
        # yaw_rate r and lateral acceleration a_y = v * r
        if abs(WHEELBASE) > 1e-6:
            r = (target_speed / WHEELBASE) * math.tan(used_steer_rad)
        else:
            r = 0.0
        ay = target_speed * r

        # Save
        times.append(t)
        xs.append(x_new)
        ys.append(y_new)
        yaws.append(yaw_new)
        speeds.append(target_speed)
        steers.append(math.degrees(used_steer_rad))
        yaw_rates.append(r)
        lat_accels.append(ay)

        # Update state for next step
        state = (x_new, y_new, yaw_new)

    return {
        't': np.array(times),
        'x': np.array(xs),
        'y': np.array(ys),
        'yaw': np.array(yaws),
        'speed': np.array(speeds),
        'steer': np.array(steers),
        'yaw_rate': np.array(yaw_rates),
        'lat_accel': np.array(lat_accels),
    }


def export_simulation_image(path, histories, show_car=False):
    """Create a multi-panel figure from histories and save to `path`.

    Panels: aerial trajectory (left), speed/steer/yaw_rate/lat_accel vs time (right).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t = histories['t']

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 3, figure=fig)

    # Aerial trajectory spans left two rows
    ax0 = fig.add_subplot(gs[:, :2])
    ax0.plot(histories['x'], histories['y'], '-b')
    ax0.scatter(histories['x'][0], histories['y'][0], c='g', label='start')
    ax0.scatter(histories['x'][-1], histories['y'][-1], c='r', label='end')
    # Draw heading arrows at start and end
    try:
        yaw0 = float(histories['yaw'][0])
        yawf = float(histories['yaw'][-1])
        arrow_len = max(0.5, np.linalg.norm([histories['x'][-1]-histories['x'][0], histories['y'][-1]-histories['y'][0]]) * 0.05)
        ax0.arrow(histories['x'][0], histories['y'][0], arrow_len * math.cos(yaw0), arrow_len * math.sin(yaw0), head_width=0.2, head_length=0.3, fc='g', ec='g')
        ax0.arrow(histories['x'][-1], histories['y'][-1], arrow_len * math.cos(yawf), arrow_len * math.sin(yawf), head_width=0.2, head_length=0.3, fc='r', ec='r')
    except Exception:
        pass
    ax0.set_aspect('equal')
    ax0.set_xlabel('X [m]')
    ax0.set_ylabel('Y [m]')
    ax0.set_title('Aerial Trajectory')
    ax0.legend()

    # Right column: four stacked plots
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.plot(t, histories['speed'], '-k')
    ax1.set_ylabel('Speed [m/s]')
    ax1.set_title('Speed')

    ax2 = fig.add_subplot(gs[1, 2])
    ax2.plot(t, histories['steer'], '-r')
    ax2.set_ylabel('Steer [deg]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Steering')

    # Small subplots below the aerial (in the middle column)
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(t, histories['yaw_rate'], '-m')
    ax3.set_title('Yaw Rate [rad/s]')
    ax3.set_xlabel('Time [s]')

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(t, histories['lat_accel'], '-c')
    ax4.set_title('Lateral Acceleration [m/s^2]')
    ax4.set_xlabel('Time [s]')

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def export_simulation_images(path, histories, show_car=False):
    """Save two images: aerial trajectory and dynamics row.

    `path` is a base path; function will write `<base>_trajectory.png` and
    `<base>_dynamics.png` (keeps extension-agnostic behavior).
    """
    import os
    base, _ = os.path.splitext(path)

    # Trajectory image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig1, ax = plt.subplots(figsize=(8, 8))
    ax.plot(histories['x'], histories['y'], '-b')
    ax.scatter(histories['x'][0], histories['y'][0], c='g', label='start')
    ax.scatter(histories['x'][-1], histories['y'][-1], c='r', label='end')
    # Draw heading arrows at start and end
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

    # Dynamics row: speed, steer, yaw_rate, lat_accel, yaw (degrees)
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
    # Yaw angle (convert to degrees for easier reading)
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
    import os, csv
    base, _ = os.path.splitext(path)
    csv_path = f"{base}_history.csv"
    keys = ['t', 'x', 'y', 'yaw', 'speed', 'steer', 'yaw_rate', 'lat_accel']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        n = len(histories['t'])
        for i in range(n):
            w.writerow([histories[k][i] for k in keys])
    return csv_path

# --- Simulation Parameters ---
SIM_DURATION = 15  # seconds
DT = 0.1  # seconds (time step)
INITIAL_STATE = [0, 0, 0]  # [x, y, yaw (radians)]
INITIAL_SPEED = 0  # m/s

# --- Vehicle Class ---
class Vehicle:
    """
    Represents an Ackermann steering vehicle with its state and kinematics.
    The state (x, y, yaw) refers to the center of the rear axle.
    """
    def __init__(self, x, y, yaw, speed, wheelbase, track_width, max_steer_angle_rad, wheel_length=0.6, wheel_width=0.5):
        self.x = x  # x-position (center of rear axle)
        self.y = y  # y-position (center of rear axle)
        self.yaw = yaw  # yaw angle (radians, heading of the vehicle)
        self.speed = speed  # speed (m/s)

        self.wheelbase = wheelbase
        self.track_width = track_width
        self.max_steer_angle_rad = max_steer_angle_rad

        # For visualization: define vehicle body and wheel dimensions
        self.body_length = wheelbase + 2* wheel_length  # Total length including overhangs
        self.body_width = track_width + wheel_width  # Total width including fenders
        self.rear_overhang = wheel_length
        self.front_overhang = self.body_length - self.wheelbase - self.rear_overhang

        self.wheel_length = wheel_length

        self.wheel_width = wheel_width

    def update(self, dt, target_speed, target_steer_angle_deg):
        """Update the vehicle state using the (refactored) kinematic helpers.

        This method now delegates the actual kinematic computation to
        `compute_kinematics` and `propagate_state` (both defined at module
        level). The split makes it easy to call the kinematics directly from
        unit tests or other tools.

        Args:
            dt (float): Time step.
            target_speed (float): Desired speed (m/s).
            target_steer_angle_deg (float): Desired steering angle (deg).

        Returns:
            float: The actual steering angle in radians used for the update
                   (after clipping) — kept for backward compatibility with the
                   rest of the code (visualization uses this).
        """
        # Update speed (direct set; no acceleration model here)
        self.speed = target_speed

        # Use the new helper to propagate state
        x_new, y_new, yaw_new, used_steer_rad = propagate_state(
            state=(self.x, self.y, self.yaw),
            speed=self.speed,
            steer_deg=target_steer_angle_deg,
            dt=dt,
            wheelbase=self.wheelbase
        )

        self.x = x_new
        self.y = y_new
        self.yaw = yaw_new

        return used_steer_rad

    def get_car_body_vertices(self):
        """
        Calculates the vertices of the car body rectangle for visualization.
        The car body is defined relative to the rear axle center (self.x, self.y)
        and oriented by self.yaw.
        Returns:
            tuple: (x_coords, y_coords) of the car body vertices.
        """
        half_width = self.body_width / 2

        # Vertices in vehicle's local coordinate system (rear axle center is (0,0))
        # (x_local, y_local)
        vertices_local = np.array([
            [-self.rear_overhang, half_width],         # Rear-left
            [self.wheelbase + self.front_overhang, half_width],  # Front-left
            [self.wheelbase + self.front_overhang, -half_width], # Front-right
            [-self.rear_overhang, -half_width],        # Rear-right
            [-self.rear_overhang, half_width]          # Close the rectangle
        ])

        # Create rotation matrix for vehicle's yaw
        rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)]
        ])

        # Rotate and translate vertices to global coordinates
        rotated_vertices = (rotation_matrix @ vertices_local.T).T
        global_vertices_x = self.x + rotated_vertices[:, 0]
        global_vertices_y = self.y + rotated_vertices[:, 1]

        return global_vertices_x, global_vertices_y

    def get_wheel_vertices(self, steer_angle_rad):
        """
        Calculates the vertices of all four wheels for visualization,
        applying Ackermann steering geometry to the front wheels.
        Args:
            steer_angle_rad (float): The steering angle of the front axle (average)
                                     as used in the bicycle model update.
        Returns:
            list: A list of (x_coords, y_coords) tuples for each wheel (FL, FR, RL, RR).
        """
        wheel_half_len = self.wheel_length / 2
        wheel_half_width = self.wheel_width / 2
        half_track = self.track_width / 2

        # Wheel positions relative to vehicle's rear axle center (0,0) in vehicle frame
        wheel_centers_local = {
            'RL': np.array([0, half_track]),
            'RR': np.array([0, -half_track]),
            'FL': np.array([self.wheelbase, half_track]),
            'FR': np.array([self.wheelbase, -half_track]),
        }

        # Calculate individual front wheel steering angles for Ackermann visualization
        delta_fl = 0
        delta_fr = 0
        if abs(steer_angle_rad) > 1e-6:
            # Calculate turning radius of the rear axle center based on the bicycle model's steer angle
            turn_radius_rear_axle = self.wheelbase / np.tan(steer_angle_rad)

            # Calculate individual wheel angles using Ackermann geometry
            # The sign of steer_angle_rad determines which side is inner/outer
            delta_fl = np.arctan(self.wheelbase / (turn_radius_rear_axle - np.sign(steer_angle_rad) * half_track))
            delta_fr = np.arctan(self.wheelbase / (turn_radius_rear_axle + np.sign(steer_angle_rad) * half_track))
        else:
            delta_fl = 0
            delta_fr = 0

        wheel_angles = {
            'RL': 0, # Rear wheels are fixed straight
            'RR': 0,
            'FL': delta_fl,
            'FR': delta_fr,
        }

        all_wheel_vertices = []

        # Vertices of a single wheel in its local coordinate system (centered at 0,0, aligned with x-axis)
        wheel_vertices_local_template = np.array([
            [-wheel_half_len, wheel_half_width],
            [wheel_half_len, wheel_half_width],
            [wheel_half_len, -wheel_half_width],
            [-wheel_half_len, -wheel_half_width],
            [-wheel_half_len, wheel_half_width] # Close the rectangle
        ])

        # Rotation matrix for the entire vehicle's yaw
        vehicle_rotation_matrix = np.array([
            [np.cos(self.yaw), -np.sin(self.yaw)],
            [np.sin(self.yaw), np.cos(self.yaw)]
        ])

        for wheel_key in ['FL', 'FR', 'RL', 'RR']: # Order for consistent plotting
            center_local = wheel_centers_local[wheel_key]
            wheel_angle = wheel_angles[wheel_key]

            # 1. Rotate wheel vertices by its individual steering angle
            wheel_rotation_matrix = np.array([
                [np.cos(wheel_angle), -np.sin(wheel_angle)],
                [np.sin(wheel_angle), np.cos(wheel_angle)]
            ])
            rotated_wheel_vertices = (wheel_rotation_matrix @ wheel_vertices_local_template.T).T

            # 2. Translate wheel vertices to its position in the vehicle's local frame
            wheel_vertices_in_vehicle_frame = rotated_wheel_vertices + center_local

            # 3. Rotate and translate wheel vertices to global coordinates
            global_wheel_vertices = (vehicle_rotation_matrix @ wheel_vertices_in_vehicle_frame.T).T + np.array([self.x, self.y])

            all_wheel_vertices.append((global_wheel_vertices[:, 0], global_wheel_vertices[:, 1]))

        return all_wheel_vertices


# --- Main Simulation Loop ---
def main():
    # Initialize vehicle object
    vehicle = Vehicle(
        x=INITIAL_STATE[0],
        y=INITIAL_STATE[1],
        yaw=INITIAL_STATE[2],
        speed=INITIAL_SPEED, # Initial speed
        wheelbase=WHEELBASE,
        track_width=TRACK_WIDTH,
        max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
    )

    # Store simulation history for plotting (grow as simulation runs)
    x_history = [vehicle.x]
    y_history = [vehicle.y]
    yaw_history = [vehicle.yaw]
    speed_history = [vehicle.speed]
    steer_angle_history_rad = [0.0]

    # Real-time control state
    current_speed_cmd = 0.0
    current_steer_cmd = 0.0
    paused = False
    held_keys = set()

    # Control tuning (per-second increments when key held)
    SPEED_INC_PER_SEC = 1.0  # m/s^2 equivalent for key hold
    STEER_INC_PER_SEC = 40.0  # deg/s when holding left/right

    # Helper to reset simulation
    def reset_sim():
        nonlocal vehicle, x_history, y_history, yaw_history, speed_history, steer_angle_history_rad, current_speed_cmd, current_steer_cmd
        vehicle.x, vehicle.y, vehicle.yaw = INITIAL_STATE
        vehicle.speed = INITIAL_SPEED
        current_speed_cmd = INITIAL_SPEED
        current_steer_cmd = 0.0
        x_history = [vehicle.x]
        y_history = [vehicle.y]
        yaw_history = [vehicle.yaw]
        speed_history = [vehicle.speed]
        steer_angle_history_rad = [0.0]

    # Key event handlers
    def on_key_press(event):
        nonlocal paused, current_speed_cmd
        key = event.key
        if key is None:
            return
        # Map arrow keys and single chars
        if key in ('left', 'right', 'up', 'down'):
            held_keys.add(key)
        elif key == ' ' or key == 'space':
            paused = not paused
        elif key == 'r':
            reset_sim()
        elif key == 'e':
            # toggle external schedule
            global external_enabled
            external_enabled = not external_enabled
        elif key == 's':
            # stop quickly
            held_keys.discard('up')
            held_keys.discard('down')
            # immediate stop
            current_speed_cmd = 0.0
            vehicle.speed = 0.0
        # ignore other keys

    def on_key_release(event):
        key = event.key
        if key in held_keys:
            held_keys.discard(key)


    # --- Visualization with Matplotlib Animation ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X Position [m]')
    ax.set_ylabel('Y Position [m]')

    # Initialize plot elements
    path_line, = ax.plot([], [], 'b-', linewidth=1.5, label='Vehicle Path')
    car_body_patch, = ax.plot([], [], 'k-', linewidth=3, label='Vehicle Body')
    # Wheels: Front-Left, Front-Right, Rear-Left, Rear-Right
    wheel_patches = [ax.plot([], [], 'k-', linewidth=2)[0] for _ in range(4)]
    
    # Text elements for dynamic title/info
    current_title = ax.set_title('')
    # On-screen status text for external schedule
    schedule_status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')

    ax.legend()

    # Set initial plot limits (will be adjusted dynamically to follow the car)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    def init():
        """
        Initialization function for the animation.
        Sets initial empty data for all plot elements.
        """
        path_line.set_data([], [])
        car_body_patch.set_data([], [])
        for wheel_patch in wheel_patches:
            wheel_patch.set_data([], [])
        current_title.set_text('')
        return [path_line, car_body_patch, *wheel_patches, current_title]

    def update(frame):
        # Each call to update represents one time-step of length DT.
        nonlocal current_speed_cmd, current_steer_cmd, paused

        # If paused, only redraw current state
        if not paused:
            # Adjust commands based on held keys
            if 'up' in held_keys:
                current_speed_cmd += SPEED_INC_PER_SEC * DT
            if 'down' in held_keys:
                current_speed_cmd -= SPEED_INC_PER_SEC * DT
            if 'left' in held_keys:
                current_steer_cmd += STEER_INC_PER_SEC * DT
            if 'right' in held_keys:
                current_steer_cmd -= STEER_INC_PER_SEC * DT

            # If an external schedule is enabled and has an entry for current time,
            # override keyboard commands with scheduled values.
            if external_enabled and external_schedule:
                t = (len(x_history) - 1) * DT
                # Find first schedule entry that covers current time
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

            # Limit steer command to allowed range
            current_steer_cmd = max(-MAX_STEER_ANGLE_DEG, min(MAX_STEER_ANGLE_DEG, current_steer_cmd))

            # Propagate vehicle state one time-step
            used_steer = vehicle.update(DT, current_speed_cmd, current_steer_cmd)

            # Append histories
            x_history.append(vehicle.x)
            y_history.append(vehicle.y)
            yaw_history.append(vehicle.yaw)
            speed_history.append(vehicle.speed)
            steer_angle_history_rad.append(used_steer)

        # Use the latest state for visualization
        idx = -1
        path_line.set_data(x_history, y_history)

        temp_vehicle = Vehicle(
            x=x_history[idx],
            y=y_history[idx],
            yaw=yaw_history[idx],
            speed=speed_history[idx],
            wheelbase=WHEELBASE,
            track_width=TRACK_WIDTH,
            max_steer_angle_rad=np.radians(MAX_STEER_ANGLE_DEG)
        )

        car_x, car_y = temp_vehicle.get_car_body_vertices()
        car_body_patch.set_data(car_x, car_y)

        wheel_vertices = temp_vehicle.get_wheel_vertices(steer_angle_history_rad[idx])
        for i, (wx, wy) in enumerate(wheel_vertices):
            wheel_patches[i].set_data(wx, wy)

        # Update title with current simulation time and vehicle speed
        current_time = (len(x_history) - 1) * DT
        current_speed = speed_history[idx]
        current_title.set_text(f'Ackermann Steering Vehicle Simulation (Time: {current_time:.1f} s, Speed: {current_speed:.2f} m/s)')

        # Update schedule status overlay
        if external_enabled and external_schedule:
            # Find active entry
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

        # Adjust plot limits to follow the car
        margin = 10 # meters, defines the visible area around the car
        ax.set_xlim(x_history[idx] - margin, x_history[idx] + margin)
        ax.set_ylim(y_history[idx] - margin, y_history[idx] + margin)

        return [path_line, car_body_patch, *wheel_patches, current_title]

    # Create the animation
    # Connect key events
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    ani = animation.FuncAnimation(
        fig, update, frames=range(1000000),
        init_func=init, blit=False, interval=DT * 1000, repeat=False
    )

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ackermann simulator (interactive)')
    parser.add_argument('--schedule', '-s', help='Path to schedule CSV to load at startup')
    parser.add_argument('--enable', action='store_true', help='Enable loaded schedule immediately')
    parser.add_argument('--export', '-x', help='Export simulation image to PATH (runs headless)')
    parser.add_argument('--duration', type=float, help='Duration (s) to use for export run (overrides schedule end)')
    args = parser.parse_args()

    # Helper: script-local directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # schedules are stored at repository root `schedules/` (one level up from script dir)
    schedules_dir = os.path.join(script_dir, 'schedules')
    default_output_dir = os.path.join(script_dir, 'output')

    # Resolve schedule path: prefer user-provided path, otherwise try schedules/ folder
    schedule_path = None
    tried_paths = []
    if args.schedule:
        # absolute path of user-provided value
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
        # Do not create outputs if the schedule failed to open; exit with non-zero
        sys.exit(1)

    if schedule_path:
        try:
            load_schedule_from_csv(schedule_path)
            print(f'Loaded schedule from {schedule_path}')
        except Exception as e:
            print(f'Failed to load schedule from {schedule_path}: {e}')

    if args.enable:
        external_enabled = True

    # If export requested, run batch simulation and exit
    if args.export:
        # Create output dir now that we know schedule succeeded (if provided)
        os.makedirs(default_output_dir, exist_ok=True)
        # If a schedule was loaded above, use external_schedule; otherwise None
        histories = batch_simulate(schedule=external_schedule if external_schedule else None, duration=args.duration)

        # Resolve export path: put files under script-level output folder by default
        export_arg = args.export
        # If user passes a bare basename, place it under default_output_dir
        if os.path.sep not in export_arg:
            export_path = os.path.join(default_output_dir, export_arg)
        else:
            # If they explicitly used the 'output' directory, route to the script-level default_output_dir
            parts = export_arg.split(os.path.sep)
            if parts[0] == 'output':
                export_path = os.path.join(default_output_dir, *parts[1:])
            else:
                export_path = export_arg

        # Ensure parent directory exists
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