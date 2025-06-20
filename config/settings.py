"""
Settings and configuration for ChipGeopolitics simulation framework.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory paths
DATA_PATHS = {
    "base": PROJECT_ROOT / "data",
    "synthetic": PROJECT_ROOT / "data" / "synthetic",
    "real": PROJECT_ROOT / "data" / "real", 
    "processed": PROJECT_ROOT / "data" / "processed",
    "outputs": PROJECT_ROOT / "data" / "outputs"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "log_dir": PROJECT_ROOT / "logs",
    "max_file_size": "10 MB",
    "retention": "1 month"
}

# Simulation settings
SIMULATION_SETTINGS = {
    # Simulation timing
    "start_year": 2025,
    "end_year": 2035,
    "time_step_months": 1,
    
    # Monte Carlo parameters
    "mc_iterations": 1000,
    "random_seed": 42,
    
    # Model parameters
    "model_accuracy_tolerance": 0.10,  # ±10%
    "convergence_threshold": 0.001,
    "max_iterations": 1000,
    
    # Performance settings
    "parallel_processing": True,
    "max_workers": 4,
    "batch_size": 100,
    
    # Data generation
    "generate_synthetic_data": True,
    "use_real_data_when_available": True,
    "data_validation_enabled": True,
    
    # Visualization
    "enable_real_time_dashboard": True,
    "dashboard_update_interval": 5,  # seconds
    "chart_resolution": "high",
    
    # Output settings
    "save_intermediate_results": True,
    "export_formats": ["csv", "json", "parquet"],
    "enable_detailed_logging": True
}

# Model weights for different components
MODEL_WEIGHTS = {
    "market_dynamics": 0.25,
    "supply_chain": 0.25,
    "geopolitical": 0.25,
    "energy_economic": 0.25
}

# Agent configuration parameters
AGENT_PARAMETERS = {
    "hyperscaler": {
        "capex_volatility": 0.20,
        "decision_frequency": 3,  # months
        "risk_tolerance": 0.15
    },
    "chip_manufacturer": {
        "capacity_utilization_target": 0.85,
        "yield_improvement_rate": 0.02,  # monthly
        "technology_investment_ratio": 0.15
    },
    "equipment_supplier": {
        "delivery_time_variance": 0.25,
        "technology_roadmap_uncertainty": 0.10,
        "market_concentration_threshold": 0.30
    },
    "nation_state": {
        "policy_change_frequency": 6,  # months
        "economic_retaliation_threshold": 0.05,
        "strategic_stockpile_target": 0.15
    }
}

def initialize_directories():
    """Create necessary directories if they don't exist."""
    for path in DATA_PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    LOGGING_CONFIG["log_dir"].mkdir(parents=True, exist_ok=True)

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "data_paths": DATA_PATHS,
        "logging": LOGGING_CONFIG,
        "simulation": SIMULATION_SETTINGS,
        "model_weights": MODEL_WEIGHTS,
        "agents": AGENT_PARAMETERS
    }

def validate_model_weights() -> bool:
    """Validate that model weights sum to 1.0."""
    total = sum(MODEL_WEIGHTS.values())
    return abs(total - 1.0) < 0.01

# Initialize directories on import
initialize_directories() 