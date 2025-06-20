"""
Constants and baseline data for ChipGeopolitics simulation.
Based on 2024 baseline metrics from the specification.
"""

# AI Data Center Metrics (2024 Baseline)
GLOBAL_ELECTRICITY_2024 = 415  # TWh
PROJECTED_ELECTRICITY_2030 = 945  # TWh
GLOBAL_DEMAND_PERCENTAGE_2024 = 1.5  # %
GLOBAL_DEMAND_PERCENTAGE_2030 = 3.0  # %

# Regional electricity consumption growth (TWh)
REGIONAL_GROWTH = {
    "US": 240,
    "China": 175,
    "Europe": 45
}

# CAPEX and Investment Data
CAPEX_2024 = 455  # Billion USD
CAPEX_GROWTH_2024 = 51  # % YoY
PROJECTED_CAPEX_2025 = 590  # Billion USD
CAPEX_GROWTH_2025 = 30  # %

# Hyperscaler investments (Billion USD)
HYPERSCALER_INVESTMENTS = {
    "Microsoft": 80,
    "Google": 75,
    "Meta": 65,
    "Amazon": 60
}

# Semiconductor Manufacturing Economics
TSMC_MARKET_SHARE_2024 = 61.7  # %
TSMC_PROJECTED_SHARE_2026 = 75.0  # %

# Process node costs (USD per wafer)
PROCESS_NODE_COSTS = {
    "28nm_taiwan": 2500,
    "28nm_smic": 1500,  # 40% reduction
    "5nm_taiwan": 9566,
    "5nm_arizona": 12724,  # 33% premium
    "2nm_estimated_min": 15000,
    "2nm_estimated_max": 20000
}

# Development costs (Million USD)
DEVELOPMENT_COSTS = {
    "28nm": 51,
    "16nm": 100,
    "5nm": 542,
    "2nm": 2000  # Estimated
}

# Fab construction cost multipliers (Taiwan = 1.0x baseline)
FAB_COST_MULTIPLIERS = {
    "Taiwan": 1.0,
    "Japan": 1.5,
    "US": 2.5,
    "Europe": 4.0
}

# Export Control Timeline
EXPORT_CONTROL_DATES = [
    "2022-10",
    "2023-10", 
    "2024-12",
    "2025-03"
]

# China localization targets
CHINA_SELF_SUFFICIENCY_2020 = 24  # %
CHINA_SELF_SUFFICIENCY_TARGET_2030 = 80  # %

# Energy Infrastructure Constraints
GRID_STRAIN_THRESHOLD_US = 10  # % consumption in states
GRID_STRAIN_THRESHOLD_IRELAND = 20  # %
AI_SERVER_POWER_MULTIPLIER = 12.5  # 10-15x higher than traditional
COOLING_OVERHEAD_PERCENTAGE = 39  # 38-40% of total consumption
RENEWABLE_TARGET_2030 = 47.5  # 45-50% clean energy mix

# Manufacturing Capacities and Constraints
ASML_EUV_ANNUAL_CAPACITY = 50  # units per year
CHINA_RARE_EARTH_PROCESSING = 60  # % global control

# Market Size Evolution (Billion USD)
TOTAL_ADDRESSABLE_MARKET = {
    "2024": 574,
    "2030": 1200
}

AI_CHIP_MARKET = {
    "2024": 45,
    "2030": 400
}

DATA_CENTER_REAL_ESTATE_ANNUAL = 200  # Billion USD
ENERGY_INFRASTRUCTURE_INVESTMENT = 500  # Billion USD renewable capacity

# Risk Probabilities (Annual %)
BLACK_SWAN_PROBABILITIES = {
    "major_natural_disaster": {
        "probability": 2.0,
        "impact_min": 50,  # Billion USD
        "impact_max": 200
    },
    "cyber_warfare": {
        "probability": 5.0,
        "capacity_disruption_min": 15,  # %
        "capacity_disruption_max": 30
    },
    "trade_war_intensification": {
        "probability": 15.0,
        "cost_increase_min": 25,  # %
        "cost_increase_max": 40
    },
    "tech_breakthrough": {
        "probability": 8.0,
        "market_disruption": "major"
    }
}

# Simulation Parameters
DEFAULT_SIMULATION_YEARS = 10  # 2025-2035
MONTE_CARLO_ITERATIONS = 1000
DEFAULT_RANDOM_SEED = 42

# Agent Counts for Synthetic Data
AGENT_COUNTS = {
    "hyperscalers": 10,
    "chip_manufacturers": 50,
    "equipment_suppliers": 100,
    "nation_states": 50,
    "total_companies": 500
}

# Model Validation Targets
MODEL_ACCURACY_TARGET = 10  # ±10% variance vs real-world data
SIMULATION_YEARS_VALIDATION = 2  # 2025-2026 projections 