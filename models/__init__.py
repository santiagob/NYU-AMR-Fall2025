"""Vehicle Models Module

Provides multiple vehicle dynamics models for simulation:
- vehicle_dynamics.DynamicBicycleModel: Full dynamics with tire slip
- vehicle_kinematics.KinematicBicycleModel: Pure kinematics with constraints
- model_factory: Factory functions for easy model switching
"""

from .vehicle_dynamics import DynamicBicycleModel, VehicleParams
from .vehicle_kinematics import KinematicBicycleModel, VehicleKinematicParams
from .model_factory import create_vehicle, get_vehicle_params, compare_models

__all__ = [
    'DynamicBicycleModel',
    'KinematicBicycleModel',
    'VehicleParams',
    'VehicleKinematicParams',
    'create_vehicle',
    'get_vehicle_params',
    'compare_models',
]
