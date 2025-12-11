#!/usr/bin/env python3
"""
Test script to validate the simplified dynamic model.
Tests lateral velocity behavior and overall stability.
"""

import numpy as np
import sys
import os

# Import from Simulator
sys.path.insert(0, os.path.dirname(__file__))
from Simulator import compute_dynamics, propagate_state_dynamic

def test_lateral_velocity_control():
    """Test that lateral velocity remains controlled during turns"""
    print("=" * 60)
    print("TEST 1: Lateral Velocity Control (Moderate 25° Turn)")
    print("=" * 60)
    
    # Initial state: moving forward at 5 m/s
    state = (0, 0, 0, 5.0, 0.0, 0.0)  # x, y, yaw, vx, vy, yaw_rate
    dt = 0.1
    
    print(f"\nInitial: vx={state[3]:.2f} m/s, vy={state[4]:.3f} m/s")
    
    # Apply moderate steering for 2 seconds
    for i in range(20):
        state = propagate_state_dynamic(state, throttle=0.0, steer_deg=25.0, dt=dt)
        if i % 5 == 0:
            print(f"t={i*dt:.1f}s: vx={state[3]:.2f} m/s, vy={state[4]:.3f} m/s, "
                  f"yaw={np.degrees(state[2]):.1f}°, yaw_rate={state[5]:.3f} rad/s")
    
    final_vy = state[4]
    print(f"\n✓ Final lateral velocity: {final_vy:.3f} m/s")
    
    if abs(final_vy) < 2.0:
        print("✓ PASS: Lateral velocity well controlled (|vy| < 2.0 m/s)")
        return True
    else:
        print("✗ WARNING: Lateral velocity still high (|vy| >= 2.0 m/s)")
        return False

def test_sharp_turn_stability():
    """Test vehicle behavior during sharp turn"""
    print("\n" + "=" * 60)
    print("TEST 2: Sharp Turn Stability (45° Steering)")
    print("=" * 60)
    
    state = (0, 0, 0, 5.0, 0.0, 0.0)
    dt = 0.1
    
    print(f"\nInitial: vx={state[3]:.2f} m/s, vy={state[4]:.3f} m/s")
    
    # Sharp turn for 2 seconds
    for i in range(20):
        state = propagate_state_dynamic(state, throttle=0.0, steer_deg=45.0, dt=dt)
        if i % 5 == 0:
            print(f"t={i*dt:.1f}s: vx={state[3]:.2f} m/s, vy={state[4]:.3f} m/s, "
                  f"yaw={np.degrees(state[2]):.1f}°")
    
    final_vy = state[4]
    print(f"\n✓ Final lateral velocity: {final_vy:.3f} m/s")
    
    if abs(final_vy) < 3.0:
        print("✓ PASS: Vehicle remains stable in sharp turn (|vy| < 3.0 m/s)")
        return True
    else:
        print("✗ WARNING: Vehicle may be unstable (|vy| >= 3.0 m/s)")
        return False

def test_straight_line_stability():
    """Test that vehicle tracks straight with no steering"""
    print("\n" + "=" * 60)
    print("TEST 3: Straight Line Stability (No Steering)")
    print("=" * 60)
    
    state = (0, 0, 0, 5.0, 0.01, 0.0)  # Small initial lateral velocity
    dt = 0.1
    
    print(f"\nInitial: vx={state[3]:.2f} m/s, vy={state[4]:.3f} m/s")
    
    # Drive straight for 2 seconds
    for i in range(20):
        state = propagate_state_dynamic(state, throttle=0.0, steer_deg=0.0, dt=dt)
        if i % 5 == 0:
            print(f"t={i*dt:.1f}s: vx={state[3]:.2f} m/s, vy={state[4]:.3f} m/s, "
                  f"yaw={np.degrees(state[2]):.1f}°")
    
    final_vy = state[4]
    print(f"\n✓ Final lateral velocity: {final_vy:.3f} m/s")
    
    if abs(final_vy) < 0.5:
        print("✓ PASS: Vehicle tracks straight, lateral velocity damped (|vy| < 0.5 m/s)")
        return True
    else:
        print("✗ FAIL: Lateral velocity not damped properly")
        return False

def compare_parameters():
    """Show parameter comparison"""
    print("\n" + "=" * 60)
    print("PARAMETER COMPARISON")
    print("=" * 60)
    print("\n| Parameter | Old Value | New Value | Change |")
    print("|-----------|-----------|-----------|---------|")
    print("| Cornering Stiffness | 15000 N/rad | 8000 N/rad | -47% (less aggressive) |")
    print("| Lateral Damping | 0 N·s/m | 2500 N·s/m | NEW (critical!) |")
    print("| Tire Model | Saturation | Linear | Simplified |")
    print("| Max Slip Angle | 12° | 15° | +25% (more tolerance) |")
    
    print("\n" + "=" * 60)
    print("KEY IMPROVEMENTS FOR REALISTIC MOVEMENT")
    print("=" * 60)
    print("""
1. REDUCED CORNERING STIFFNESS (15000 → 8000 N/rad)
   - Less aggressive lateral response to steering
   - More gradual turn-in behavior
   - Better matches passenger car feel

2. ADDED LATERAL DAMPING (2500 N·s/m)
   - **Critical change**: Directly opposes lateral velocity buildup
   - Prevents excessive drifting and sliding
   - Represents aerodynamic side forces and tire scrub
   - Formula: F_damping = -2500 * vy

3. SIMPLIFIED TIRE MODEL
   - Removed complex saturation logic
   - Pure linear model with gentle clamping
   - More predictable behavior

4. INCREASED SLIP TOLERANCE (12° → 15°)
   - Allows more steering before saturation
   - Smoother handling at moderate angles
""")

if __name__ == "__main__":
    print("\nTesting Simplified Dynamic Model")
    print("=" * 60)
    
    results = []
    results.append(test_lateral_velocity_control())
    results.append(test_sharp_turn_stability())
    results.append(test_straight_line_stability())
    
    compare_parameters()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The simplified dynamics should show:")
        print("  - Less pronounced lateral movement")
        print("  - Reduced drifting and sliding")
        print("  - More realistic car-like behavior")
        print("\nRegenerate animations to see the improvement!")
    else:
        print(f"\n⚠ {total - passed} test(s) need attention")
        print("Consider adjusting LATERAL_DAMPING_COEFF or CORNERING_STIFFNESS")
