#!/usr/bin/env python3
"""
Main script to run the ChipGeopolitics simulation.
Demonstrates the complete simulation workflow from setup to analysis.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import get_config, SIMULATION_SETTINGS
from config.constants import AGENT_COUNTS, DEFAULT_SIMULATION_YEARS
# Note: Import ChipGeopoliticsModel after agent classes are created
from src.data_generation.company_profiles import CompanyProfileGenerator
from src.core.monte_carlo import MonteCarloEngine, ScenarioDefinition, DistributionParams, DistributionType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_simulation_environment():
    """Setup the simulation environment and generate synthetic data."""
    logger.info("Setting up simulation environment...")
    
    # Get configuration
    config = get_config()
    
    # Generate company profiles
    logger.info("Generating synthetic company profiles...")
    profile_generator = CompanyProfileGenerator(random_seed=42)
    
    # Generate profiles for each agent type
    hyperscaler_profiles = profile_generator.generate_hyperscaler_profiles(
        AGENT_COUNTS['hyperscalers']
    )
    chip_manufacturer_profiles = profile_generator.generate_chip_manufacturer_profiles(
        AGENT_COUNTS['chip_manufacturers']
    )
    equipment_supplier_profiles = profile_generator.generate_equipment_supplier_profiles(
        AGENT_COUNTS['equipment_suppliers']
    )
    nation_state_profiles = profile_generator.generate_nation_state_profiles(
        AGENT_COUNTS['nation_states']
    )
    
    # Save profiles for reference
    output_dir = Path("data/synthetic")
    profile_generator.save_profiles_to_json(
        hyperscaler_profiles, "hyperscaler_profiles.json", output_dir
    )
    profile_generator.save_profiles_to_json(
        chip_manufacturer_profiles, "chip_manufacturer_profiles.json", output_dir
    )
    profile_generator.save_profiles_to_json(
        equipment_supplier_profiles, "equipment_supplier_profiles.json", output_dir
    )
    profile_generator.save_profiles_to_json(
        nation_state_profiles, "nation_state_profiles.json", output_dir
    )
    
    logger.info(f"Generated {len(hyperscaler_profiles)} hyperscaler profiles")
    logger.info(f"Generated {len(chip_manufacturer_profiles)} chip manufacturer profiles")
    logger.info(f"Generated {len(equipment_supplier_profiles)} equipment supplier profiles")
    logger.info(f"Generated {len(nation_state_profiles)} nation-state profiles")
    
    return config


def create_monte_carlo_scenarios():
    """Create Monte Carlo scenarios for uncertainty analysis."""
    logger.info("Creating Monte Carlo scenarios...")
    
    monte_carlo = MonteCarloEngine(random_seed=42)
    
    # Baseline scenario
    baseline_scenario = ScenarioDefinition(
        name="baseline",
        description="Baseline scenario with current trends",
        variable_distributions={
            "chip_demand_growth": DistributionParams(
                DistributionType.NORMAL, {"mean": 0.30, "std": 0.05}
            ),
            "geopolitical_tension": DistributionParams(
                DistributionType.BETA, {"alpha": 2, "beta": 3}
            ),
            "supply_chain_disruption": DistributionParams(
                DistributionType.UNIFORM, {"low": 0.02, "high": 0.08}
            ),
            "energy_costs": DistributionParams(
                DistributionType.LOGNORMAL, {"mean": 0.1, "sigma": 0.2}
            )
        },
        probability=0.5
    )
    
    # Trade war escalation scenario
    trade_war_scenario = ScenarioDefinition(
        name="trade_war_escalation",
        description="Scenario with escalating trade tensions",
        variable_distributions={
            "chip_demand_growth": DistributionParams(
                DistributionType.NORMAL, {"mean": 0.20, "std": 0.08}
            ),
            "geopolitical_tension": DistributionParams(
                DistributionType.BETA, {"alpha": 4, "beta": 2}
            ),
            "supply_chain_disruption": DistributionParams(
                DistributionType.UNIFORM, {"low": 0.10, "high": 0.25}
            ),
            "export_control_intensity": DistributionParams(
                DistributionType.UNIFORM, {"low": 0.6, "high": 0.9}
            )
        },
        probability=0.25,
        conditions={"trade_war_active": True}
    )
    
    # Technology breakthrough scenario
    tech_breakthrough_scenario = ScenarioDefinition(
        name="technology_breakthrough",
        description="Scenario with major technological advancement",
        variable_distributions={
            "chip_demand_growth": DistributionParams(
                DistributionType.NORMAL, {"mean": 0.45, "std": 0.10}
            ),
            "innovation_rate": DistributionParams(
                DistributionType.BETA, {"alpha": 4, "beta": 2}
            ),
            "production_efficiency": DistributionParams(
                DistributionType.UNIFORM, {"low": 1.2, "high": 1.8}
            )
        },
        probability=0.15,
        conditions={"breakthrough_active": True}
    )
    
    # Supply chain crisis scenario
    supply_crisis_scenario = ScenarioDefinition(
        name="supply_chain_crisis",
        description="Major supply chain disruption scenario",
        variable_distributions={
            "supply_chain_disruption": DistributionParams(
                DistributionType.UNIFORM, {"low": 0.30, "high": 0.60}
            ),
            "chip_prices": DistributionParams(
                DistributionType.NORMAL, {"mean": 1.5, "std": 0.3}
            ),
            "delivery_delays": DistributionParams(
                DistributionType.GAMMA, {"shape": 2, "scale": 3}
            )
        },
        probability=0.10,
        conditions={"supply_crisis_active": True}
    )
    
    # Add scenarios to Monte Carlo engine
    monte_carlo.add_scenario(baseline_scenario)
    monte_carlo.add_scenario(trade_war_scenario)
    monte_carlo.add_scenario(tech_breakthrough_scenario)
    monte_carlo.add_scenario(supply_crisis_scenario)
    
    return monte_carlo


def run_base_simulation():
    """Run the base deterministic simulation."""
    logger.info("Running base simulation...")
    
    # Create simulation model
    model = ChipGeopoliticsModel(
        start_year=2025,
        end_year=2030,  # 5-year simulation for demo
        agent_counts=AGENT_COUNTS,
        random_seed=42
    )
    
    # Run simulation
    simulation_steps = (2030 - 2025) * 12  # Monthly steps
    results = model.run_simulation(steps=simulation_steps)
    
    # Get agent data
    agent_data = model.get_agent_data()
    
    # Save results
    output_dir = Path("data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_dir / "base_simulation_results.csv", index=False)
    agent_data.to_csv(output_dir / "base_agent_data.csv", index=False)
    
    # Get supply chain analysis
    supply_chain_analysis = model.get_supply_chain_analysis()
    logger.info(f"Supply chain analysis: {supply_chain_analysis}")
    
    return model, results, agent_data


def run_monte_carlo_analysis(monte_carlo: MonteCarloEngine):
    """Run Monte Carlo uncertainty analysis."""
    logger.info("Running Monte Carlo analysis...")
    
    def simulation_model_function(params: dict) -> dict:
        """Model function for Monte Carlo analysis."""
        # Create model with sampled parameters
        initial_conditions = {
            "chip_demand_growth": params.get("chip_demand_growth", 0.30),
            "geopolitical_tension": params.get("geopolitical_tension", 0.1),
            "supply_chain_disruption": params.get("supply_chain_disruption", 0.05)
        }
        
        model = ChipGeopoliticsModel(
            start_year=2025,
            end_year=2027,  # Shorter for MC analysis
            agent_counts={k: max(1, v//4) for k, v in AGENT_COUNTS.items()},  # Smaller for speed
            initial_conditions=initial_conditions,
            random_seed=np.random.randint(0, 10000)
        )
        
        # Run simulation
        steps = (2027 - 2025) * 12
        model.run_simulation(steps=steps)
        
        # Return key metrics
        return {
            "total_chip_production": model.metrics.total_chip_production,
            "global_electricity_consumption": model.metrics.global_electricity_consumption,
            "supply_chain_resilience": model.metrics.supply_chain_resilience_index,
            "economic_welfare": model.metrics.economic_welfare_index,
            "carbon_emissions": model.metrics.carbon_emissions,
            "geopolitical_stability": model.metrics.geopolitical_stability_index
        }
    
    # Run Monte Carlo simulation
    mc_results = monte_carlo.run_simulation(
        model_function=simulation_model_function,
        n_iterations=100,  # Reduced for demo
        parallel=True,
        max_workers=2
    )
    
    # Generate comprehensive analysis report
    target_columns = [
        "total_chip_production", "global_electricity_consumption",
        "supply_chain_resilience", "economic_welfare", "carbon_emissions"
    ]
    
    analysis_report = monte_carlo.generate_scenario_report(mc_results, target_columns)
    
    # Save Monte Carlo results
    output_dir = Path("data/outputs")
    mc_results.to_csv(output_dir / "monte_carlo_results.csv", index=False)
    
    # Save analysis report
    import json
    with open(output_dir / "monte_carlo_analysis.json", 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)
    
    logger.info("Monte Carlo analysis completed")
    return mc_results, analysis_report


def generate_summary_report(base_results: pd.DataFrame, 
                          agent_data: pd.DataFrame,
                          mc_results: pd.DataFrame,
                          mc_analysis: dict):
    """Generate a summary report of simulation results."""
    logger.info("Generating summary report...")
    
    # Create summary statistics
    summary = {
        "simulation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "base_simulation_steps": len(base_results),
            "monte_carlo_iterations": len(mc_results),
            "agent_count": len(agent_data['AgentID'].unique()) if 'AgentID' in agent_data.columns else 0
        },
        "key_findings": {
            "final_chip_production": base_results["Total_Chip_Production"].iloc[-1] if len(base_results) > 0 else 0,
            "final_electricity_consumption": base_results["Global_Electricity_Consumption"].iloc[-1] if len(base_results) > 0 else 0,
            "average_supply_chain_resilience": base_results["Supply_Chain_Resilience"].mean() if len(base_results) > 0 else 0,
            "geopolitical_tension_trend": "increasing" if len(base_results) > 10 and base_results["Geopolitical_Tension"].iloc[-1] > base_results["Geopolitical_Tension"].iloc[0] else "stable"
        },
        "risk_assessment": mc_analysis.get("risk_metrics", {}),
        "scenario_comparison": mc_analysis.get("scenario_comparison", {}),
        "sensitivity_analysis": mc_analysis.get("sensitivity_analysis", {})
    }
    
    # Save summary report
    output_dir = Path("data/outputs")
    with open(output_dir / "simulation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print key findings
    print("\n" + "="*60)
    print("CHIPGEOPOLITICS SIMULATION SUMMARY")
    print("="*60)
    print(f"Simulation completed at: {summary['simulation_metadata']['timestamp']}")
    print(f"Base simulation steps: {summary['simulation_metadata']['base_simulation_steps']}")
    print(f"Monte Carlo iterations: {summary['simulation_metadata']['monte_carlo_iterations']}")
    print("\nKey Findings:")
    for key, value in summary['key_findings'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print("="*60)
    
    return summary


def main():
    """Main simulation execution function."""
    logger.info("Starting ChipGeopolitics simulation...")
    
    try:
        # Setup environment
        config = setup_simulation_environment()
        
        # Create Monte Carlo scenarios
        monte_carlo = create_monte_carlo_scenarios()
        
        # Run base simulation
        model, base_results, agent_data = run_base_simulation()
        
        # Run Monte Carlo analysis
        mc_results, mc_analysis = run_monte_carlo_analysis(monte_carlo)
        
        # Generate summary report
        summary = generate_summary_report(base_results, agent_data, mc_results, mc_analysis)
        
        logger.info("Simulation completed successfully!")
        
        return {
            "model": model,
            "base_results": base_results,
            "agent_data": agent_data,
            "monte_carlo_results": mc_results,
            "analysis": mc_analysis,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    # Ensure directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    
    # Run simulation
    results = main()
    print(f"\nSimulation results saved to: {Path('data/outputs').absolute()}")
    print(f"Logs saved to: {Path('logs').absolute()}") 