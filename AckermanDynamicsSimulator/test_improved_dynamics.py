#!/usr/bin/env python3
"""
Test script to demonstrate the improved dynamic model
Shows the difference between old drifty behavior and new realistic behavior
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

# Test the new dynamics
sys.path.insert(0, '/Users/santiagobernheim/Documents/Projects/NYU/NYU-AMR-Fall2025/AckermanDynamicsSimulator')
from Simulator import propagate_state_dynamic

# --- Test Case 1: Gentle turn at constant speed ---
print("=" * 70)
print("TEST 1: Gentle Turn (5 m/s, 20° steering)")
print("=" * 70)

state = [0, 0, 0, 5.0, 0, 0]  # Start moving at 5 m/s in +x direction, no lateral velocity
for i in range(20):  # 2 seconds
    state = propagate_state_dynamic(state, throttle=0.0, steer_deg=20, dt=0.1)
    if i % 5 == 0:
        print(f"  t={i*0.1:.1f}s: pos=({state[0]:6.2f}, {state[1]:6.2f}), "
              f"v=({state[3]:5.2f}, {state[4]:5.2f}) m/s, yaw={np.degrees(state[2]):6.1f}°")

print(f"\nFinal lateral velocity: {state[4]:.3f} m/s")
print(f"  → Should be small (<1 m/s) - vehicle should NOT be heavily drifting")

# --- Test Case 2: Sharp turn ---
print("\n" + "=" * 70)
print("TEST 2: Sharp Turn (5 m/s, 45° steering)")
print("=" * 70)

state = [0, 0, 0, 5.0, 0, 0]
for i in range(30):  # 3 seconds
    state = propagate_state_dynamic(state, throttle=0.0, steer_deg=45, dt=0.1)
    if i % 10 == 0:
        print(f"  t={i*0.1:.1f}s: pos=({state[0]:6.2f}, {state[1]:6.2f}), "
              f"v=({state[3]:5.2f}, {state[4]:5.2f}) m/s, yaw={np.degrees(state[2]):6.1f}°")

print(f"\nFinal lateral velocity: {state[4]:.3f} m/s")
print(f"  → May be higher (~1-2 m/s) but should stabilize, not diverge")

# --- Test Case 3: Speed and steering stability ---
print("\n" + "=" * 70)
print("TEST 3: Constant Speed Control (maintain 5 m/s)")
print("=" * 70)

state = [0, 0, 0, 5.0, 0, 0]
throttle_cmds = [0.0] * 10 + [0.1] * 10 + [0.0] * 20  # Brief acceleration, then coast

for i, throttle in enumerate(throttle_cmds):
    state = propagate_state_dynamic(state, throttle=throttle, steer_deg=10, dt=0.1)
    if i % 10 == 0:
        print(f"  t={i*0.1:.1f}s: vx={state[3]:5.2f} m/s, vy={state[4]:5.2f} m/s, "
              f"yaw_rate={state[5]:5.3f} rad/s")

print(f"\nFinal vx: {state[3]:.3f} m/s (should decline due to drag)")
print(f"Final vy: {state[4]:.3f} m/s (should remain small)")

print("\n" + "=" * 70)
print("SUMMARY OF IMPROVEMENTS")
print("=" * 70)
print("""
1. CORNERING STIFFNESS: Increased from -600 N/rad to +4000/+3500 N/rad
   → More realistic tire grip (matches standard road tires)
   → Negative values were physically incorrect (reversed force direction)

2. SLIP ANGLE SATURATION: Added 15° maximum slip angle
   → Tires naturally reduce grip when slipping excessively
   → Prevents unrealistic lateral forces at extreme angles

3. IMPROVED SLIP ANGLE CALCULATION: Changed from linear to arctan2 formula
   → More accurate at high speeds and large slip angles
   → Matches standard vehicle dynamics literature

4. BETTER LATERAL DYNAMICS: Improved force balance equations
   → Removed unrealistic vertical load transfer simplification
   → Better centripetal acceleration handling
   
5. HIGHER ROLLING RESISTANCE: 0.01 → 0.015
   → Better energy dissipation at low speeds
   → Prevents unnecessary low-speed drifting

EXPECTED RESULT:
- Vehicle should track curves smoothly WITHOUT excessive side slip
- Lateral velocity (vy) should remain much smaller than longitudinal (vx)
- Yaw rate should be proportional to steering and speed
- No continuous drifting or unrealistic behavior
""")
