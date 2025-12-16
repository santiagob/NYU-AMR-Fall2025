import argparse
import math
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from models.vehicle_kinematics import KinematicBicycleModel, VehicleKinematicParams


def load_schedule(path, dt):
    """Load a schedule CSV (start,end,speed,steer) into expanded per-dt steps."""
    entries = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
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
                n_steps = int(math.ceil(span / dt))
                for i in range(n_steps):
                    t0 = start + i * dt
                    t1 = min(end, t0 + dt)
                    t0 = round(t0, 10)
                    t1 = round(t1, 10)
                    speed = v0 + accel * (t0 - start)
                    entries.append({'start': t0, 'end': t1, 'speed': speed, 'steer': steer})
            else:
                speed = float(row.get('speed', 0.0)) if row.get('speed') not in (None, '') else 0.0
                entries.append({'start': start, 'end': end, 'speed': speed, 'steer': steer})
    return entries


def get_active_schedule(schedule, t):
    for entry in schedule:
        if entry['start'] <= t < entry['end']:
            return entry
    return None


def body_and_wheels(x, y, yaw, wheelbase, track, steer_rad):
    """Compute simple rectangle for body and four wheel line segments."""
    # Body rectangle corners relative to rear axle
    half_w = track / 2.0
    L = wheelbase
    body = np.array([
        [0.2 * L,  half_w],
        [0.2 * L, -half_w],
        [-0.2 * L, -half_w],
        [-0.2 * L,  half_w],
        [0.2 * L,  half_w]
    ])
    # Wheel centers
    front_x = L
    rear_x = 0
    wheel_len = 0.4
    wheel_w = 0.15
    wheels = {
        'FL': np.array([[wheel_len/2,  wheel_w/2], [ -wheel_len/2,  wheel_w/2], [ -wheel_len/2, -wheel_w/2], [ wheel_len/2, -wheel_w/2], [wheel_len/2, wheel_w/2]]) + [front_x,  half_w],
        'FR': np.array([[wheel_len/2,  wheel_w/2], [ -wheel_len/2,  wheel_w/2], [ -wheel_len/2, -wheel_w/2], [ wheel_len/2, -wheel_w/2], [wheel_len/2, wheel_w/2]]) + [front_x, -half_w],
        'RL': np.array([[wheel_len/2,  wheel_w/2], [ -wheel_len/2,  wheel_w/2], [ -wheel_len/2, -wheel_w/2], [ wheel_len/2, -wheel_w/2], [wheel_len/2, wheel_w/2]]) + [rear_x,  half_w],
        'RR': np.array([[wheel_len/2,  wheel_w/2], [ -wheel_len/2,  wheel_w/2], [ -wheel_len/2, -wheel_w/2], [ wheel_len/2, -wheel_w/2], [wheel_len/2, wheel_w/2]]) + [rear_x, -half_w],
    }
    # Rotate
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    body_xy = (R @ body.T).T + np.array([x, y])
    wheels_xy = {}
    for key, poly in wheels.items():
        local = poly.copy()
        if key in ('FL', 'FR'):
            # steer front wheels
            cs, ss = math.cos(steer_rad), math.sin(steer_rad)
            Rw = np.array([[cs, -ss], [ss, cs]])
            local = (Rw @ (poly - poly[0]).T).T + poly[0]
        wheels_xy[key] = (R @ local.T).T + np.array([x, y])
    return body_xy, wheels_xy


def run_keyboard(model_type='dynamic', dt=0.05, duration=60.0, schedule_path=None):
    if model_type == 'dynamic':
        vehicle = DynamicBicycleModel()
        max_steer = VehicleParams.MAX_STEER
        wheelbase = VehicleParams.WHEELBASE
        track = VehicleParams.TRACK_WIDTH
    else:
        vehicle = KinematicBicycleModel()
        max_steer = VehicleKinematicParams.MAX_STEER
        wheelbase = VehicleKinematicParams.WHEELBASE
        track = VehicleKinematicParams.TRACK_WIDTH

    schedule = load_schedule(schedule_path, dt) if schedule_path else []

    throttle_cmd = 0.0
    steer_cmd_deg = 0.0
    held = set()
    paused = False

    xs, ys, yaws, speeds, steers = [], [], [], [], []

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Keyboard Control ({model_type})")
    path_line, = ax.plot([], [], 'b-', lw=1.5, label='Path')
    body_line, = ax.plot([], [], 'k-', lw=2, label='Body')
    wheel_lines = {k: ax.plot([], [], 'k-', lw=1.5)[0] for k in ['FL','FR','RL','RR']}
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
    ax.legend(loc='upper right')

    def reset():
        nonlocal vehicle, throttle_cmd, steer_cmd_deg
        vehicle = DynamicBicycleModel() if model_type == 'dynamic' else KinematicBicycleModel()
        throttle_cmd = 0.0
        steer_cmd_deg = 0.0
        xs.clear(); ys.clear(); yaws.clear(); speeds.clear(); steers.clear()

    def on_press(event):
        nonlocal paused
        k = event.key
        if k in ('up','down','left','right'):
            held.add(k)
        elif k == ' ':  # pause
            paused = not paused
        elif k == 'r':
            reset()
        elif k == 's':
            held.discard('up'); held.discard('down')
            throttle_cmd = 0.0
        elif k == 'escape':
            plt.close(fig)

    def on_release(event):
        k = event.key
        if k in held:
            held.discard(k)

    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('key_release_event', on_release)

    def step():
        nonlocal throttle_cmd, steer_cmd_deg
        if 'up' in held:
            throttle_cmd = min(1.0, throttle_cmd + 0.5*dt)
        if 'down' in held:
            throttle_cmd = max(-1.0, throttle_cmd - 0.5*dt)
        if 'left' in held:
            steer_cmd_deg = min(60.0, steer_cmd_deg + 60*dt)
        if 'right' in held:
            steer_cmd_deg = max(-60.0, steer_cmd_deg - 60*dt)
        # apply schedule override
        t = len(xs) * dt
        active = get_active_schedule(schedule, t)
        if active:
            desired_speed = active.get('speed', 0.0)
            steer_cmd_deg = active.get('steer', steer_cmd_deg)
            # crude speed PI -> throttle
            current_speed = math.hypot(vehicle.vx, getattr(vehicle, 'vy', 0.0)) if hasattr(vehicle, 'vx') else 0.0
            err = desired_speed - current_speed
            throttle_cmd = np.clip(0.2*err, -1.0, 1.0)
        steer_rad = np.clip(math.radians(steer_cmd_deg), -max_steer, max_steer)
        vehicle.update(throttle_cmd, steer_rad, dt)
        xs.append(vehicle.x); ys.append(vehicle.y); yaws.append(vehicle.yaw)
        speeds.append(math.hypot(getattr(vehicle, 'vx', 0.0), getattr(vehicle, 'vy', 0.0)))
        steers.append(steer_rad)

    def init_anim():
        path_line.set_data([], [])
        body_line.set_data([], [])
        for wl in wheel_lines.values():
            wl.set_data([], [])
        return [path_line, body_line, *wheel_lines.values()]

    def update_anim(frame):
        if not paused:
            step()
        if xs:
            path_line.set_data(xs, ys)
            body_xy, wheels_xy = body_and_wheels(xs[-1], ys[-1], yaws[-1], wheelbase, track, steers[-1] if steers else 0.0)
            body_line.set_data(body_xy[:,0], body_xy[:,1])
            for k, wl in wheel_lines.items():
                wxy = wheels_xy[k]
                wl.set_data(wxy[:,0], wxy[:,1])
            ax.set_xlim(xs[-1]-10, xs[-1]+10)
            ax.set_ylim(ys[-1]-10, ys[-1]+10)
            status_text.set_text(f"t={len(xs)*dt:.1f}s | v={speeds[-1]:.2f} m/s | steer={math.degrees(steers[-1]):.1f} deg")
        return [path_line, body_line, *wheel_lines.values(), status_text]

    anim = animation.FuncAnimation(fig, update_anim, init_func=init_anim, interval=50, blit=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Keyboard-controlled Ackermann simulator (dynamic or kinematic)")
    parser.add_argument('--model', choices=['dynamic','kinematic'], default='dynamic', help='Model type to run')
    parser.add_argument('--dt', type=float, default=0.05, help='Simulation timestep (s)')
    parser.add_argument('--duration', type=float, default=60.0, help='Simulation duration (s) -- currently unused, animation runs until closed')
    parser.add_argument('--schedule', type=Path, default=None, help='Optional CSV schedule with start,end,speed,steer columns')
    args = parser.parse_args()

    run_keyboard(model_type=args.model, dt=args.dt, duration=args.duration, schedule_path=args.schedule)


if __name__ == "__main__":
    main()
