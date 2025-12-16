"""
Vehicle Model Factory
=====================

Provides a unified interface for switching between different vehicle models.
This allows you to easily compare and switch between:
- DynamicBicycleModel: Full dynamics with tire slip angles
- KinematicBicycleModel: Pure kinematics with rear-axle no-slip constraint

Usage:
    vehicle = create_vehicle("kinematic", x=0, y=0, yaw=0)
    # or
    vehicle = create_vehicle("dynamic", x=0, y=0, yaw=0)
"""

from models.vehicle_dynamics import DynamicBicycleModel, VehicleParams
from models.vehicle_kinematics import KinematicBicycleModel, VehicleKinematicParams


def create_vehicle(model_type="kinematic", x=0, y=0, yaw=0, vx=0, vy=0, r=0):
    """
    Factory function to create a vehicle model.
    
    Args:
        model_type: "kinematic" or "dynamic"
        x, y, yaw: Initial position and heading
        vx, vy, r: Initial velocities (optional)
    
    Returns:
        Vehicle model instance with unified interface
    
    Example:
        # Create kinematic model
        vehicle = create_vehicle("kinematic", x=5, y=5, yaw=0)
        
        # Or dynamic model with same interface
        vehicle = create_vehicle("dynamic", x=5, y=5, yaw=0)
        
        # Update in simulation loop
        vehicle.update(throttle=0.5, steer_rad=0.1, dt=0.1)
        print(f"Position: ({vehicle.x}, {vehicle.y})")
    """
    if model_type.lower() == "kinematic":
        return KinematicBicycleModel(x=x, y=y, yaw=yaw, vx=vx, vy=vy, r=r)
    elif model_type.lower() == "dynamic":
        return DynamicBicycleModel(x=x, y=y, yaw=yaw, vx=vx, vy=vy, r=r)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'kinematic' or 'dynamic'")


def get_vehicle_params(model_type="kinematic"):
    """Get vehicle parameters for a specific model"""
    if model_type.lower() == "kinematic":
        return VehicleKinematicParams()
    elif model_type.lower() == "dynamic":
        return VehicleParams()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compare_models():
    """Quick comparison of model parameters"""
    print("\n" + "="*60)
    print("VEHICLE MODEL COMPARISON")
    print("="*60)
    
    dynamic_params = VehicleParams()
    kinematic_params = VehicleKinematicParams()
    
    print(f"\nWheel base:        {dynamic_params.WHEELBASE} m")
    print(f"Track width:       {dynamic_params.TRACK_WIDTH} m")
    print(f"Mass:              {dynamic_params.MASS} kg")
    print(f"Yaw inertia:       {dynamic_params.YAW_INERTIA} kg·m²")
    print(f"CG to front:       {dynamic_params.CG_TO_FRONT} m")
    print(f"CG to rear:        {dynamic_params.CG_TO_REAR} m")
    
    print("\n" + "-"*60)
    print("KINEMATIC MODEL")
    print("-"*60)
    print("✓ Rear axle no-slip constraint: vy = CG_TO_REAR * r")
    print("✓ Yaw rate from kinematics:     r = (vx/L) * tan(δ)")
    print("✓ Simple speed control:         Kp = 0.5")
    print("✓ Fast and stable")
    print("✓ No drift or tire dynamics")
    
    print("\n" + "-"*60)
    print("DYNAMIC MODEL")
    print("-"*60)
    print("✓ Tire slip angles: α_f, α_r")
    print("✓ Tire forces: F = C_α * α")
    print("✓ Complex force balance")
    print("✓ Aerodynamic drag and rolling resistance")
    print("✓ More realistic but requires tuning")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    compare_models()
