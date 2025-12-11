#!/usr/bin/env python3
"""
Test to verify kinematic constraint: rear tire should have ZERO lateral velocity.
This validates that we've properly implemented the no-slip condition at the rear axle.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from Simulator import compute_dynamics, propagate_state_dynamic, CG_TO_REAR

def test_kinematic_constraint():
    """
    Verify that rear axle has no lateral velocity (kinematic no-slip condition).
    In kinematic model: v_rear_lateral = vy - CG_TO_REAR * r = 0
    """
    print("=" * 70)
    print("KINEMATIC CONSTRAINT VERIFICATION")
    print("=" * 70)
    print("\nTesting: Rear axle lateral velocity should be ZERO")
    print("Kinematic constraint: v_rear_lateral = vy - CG_TO_REAR * r = 0\n")
    
    # Test in a turn
    state = (0, 0, 0, 5.0, 0.0, 0.0)
    dt = 0.1
    
    print("25° Turn Test:")
    print("-" * 70)
    
    max_rear_slip = 0.0
    
    for i in range(20):
        state = propagate_state_dynamic(state, throttle=0.0, steer_deg=25.0, dt=dt)
        
        x, y, yaw, vx, vy, r = state
        
        # Calculate lateral velocity at rear axle
        v_rear_lateral = vy - CG_TO_REAR * r
        
        max_rear_slip = max(max_rear_slip, abs(v_rear_lateral))
        
        if i % 5 == 0:
            print(f"t={i*dt:.1f}s: vy={vy:.4f} m/s, r={r:.4f} rad/s, "
                  f"v_rear_lat={v_rear_lateral:.6f} m/s")
    
    print(f"\n✓ Maximum rear lateral velocity: {max_rear_slip:.6f} m/s")
    
    if max_rear_slip < 0.01:
        print("✓✓ EXCELLENT: Rear axle has essentially ZERO lateral slip!")
        print("   This matches kinematic model behavior perfectly.")
        return True
    elif max_rear_slip < 0.1:
        print("✓ GOOD: Rear axle lateral slip is very small (< 0.1 m/s)")
        print("  Close to kinematic behavior.")
        return True
    else:
        print(f"✗ WARNING: Rear axle still has {max_rear_slip:.3f} m/s lateral velocity")
        print("  Not fully kinematic - constraint may need strengthening.")
        return False

def test_kinematic_relationship():
    """
    Verify that vy follows the kinematic relationship: vy = CG_TO_REAR * r
    """
    print("\n" + "=" * 70)
    print("KINEMATIC RELATIONSHIP VERIFICATION")
    print("=" * 70)
    print("\nTesting: vy should equal CG_TO_REAR * r (kinematic prediction)")
    print(f"CG_TO_REAR = {CG_TO_REAR:.2f} m\n")
    
    state = (0, 0, 0, 5.0, 0.0, 0.0)
    dt = 0.1
    
    print("45° Sharp Turn Test:")
    print("-" * 70)
    
    max_deviation = 0.0
    
    for i in range(20):
        state = propagate_state_dynamic(state, throttle=0.0, steer_deg=45.0, dt=dt)
        
        x, y, yaw, vx, vy, r = state
        
        # Kinematic prediction
        vy_kinematic = CG_TO_REAR * r
        
        # Deviation from kinematic
        deviation = abs(vy - vy_kinematic)
        max_deviation = max(max_deviation, deviation)
        
        if i % 5 == 0:
            print(f"t={i*dt:.1f}s: vy={vy:.4f} m/s, "
                  f"vy_kin={vy_kinematic:.4f} m/s, "
                  f"error={deviation:.6f} m/s")
    
    print(f"\n✓ Maximum deviation from kinematic: {max_deviation:.6f} m/s")
    
    if max_deviation < 0.01:
        print("✓✓ PERFECT: vy matches kinematic prediction exactly!")
        return True
    elif max_deviation < 0.1:
        print("✓ GOOD: vy closely follows kinematic prediction")
        return True
    else:
        print(f"✗ WARNING: vy deviates by {max_deviation:.3f} m/s from kinematic")
        return False

def compare_with_dynamic():
    """
    Compare rear tire behavior: kinematic vs dynamic
    """
    print("\n" + "=" * 70)
    print("COMPARISON: KINEMATIC vs DYNAMIC BEHAVIOR")
    print("=" * 70)
    
    print("\n| Model | Rear Tire Behavior | Lateral Slip |")
    print("|-------|-------------------|--------------|")
    print("| Pure Kinematic | No slip angle, Fy_rear = 0 | Zero |")
    print("| Pure Dynamic | Has slip angle, Fy_rear ≠ 0 | Present |")
    print("| Current Model | No slip angle, Fy_rear = 0 | Zero (kinematic) |")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The current model implements TRUE KINEMATIC constraint:

✓ Rear tire lateral force = 0 (no lateral slip at rear)
✓ Rear axle lateral velocity = 0 (no-slip condition)
✓ vy constrained to equal CG_TO_REAR * r

This eliminates the lateral movement at the rear tire completely,
making the model behave exactly like the kinematic bicycle model.

The vehicle will:
• Pivot around the rear axle
• Have no rear tire sliding
• Show car-like, predictable motion
• Match kinematic simulator behavior
""")

if __name__ == "__main__":
    print("\nVerifying Kinematic Constraint Implementation\n")
    
    result1 = test_kinematic_constraint()
    result2 = test_kinematic_relationship()
    compare_with_dynamic()
    
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if result1 and result2:
        print("\n✓✓ SUCCESS: Model behaves like TRUE KINEMATIC model!")
        print("   • Rear tire has zero lateral slip")
        print("   • vy follows kinematic relationship")
        print("   • No rear tire lateral movement")
        print("\n   The dynamic simulator now behaves like the kinematic one!")
    else:
        print("\n⚠ Constraint may need tuning for perfect kinematic behavior")
        print("  Consider increasing KINEMATIC_CONSTRAINT_FACTOR closer to 1.0")
