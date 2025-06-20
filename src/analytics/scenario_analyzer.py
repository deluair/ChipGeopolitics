"""
Scenario Analyzer for ChipGeopolitics Simulation Framework

Comprehensive scenario comparison, sensitivity analysis, and strategic
planning capabilities for semiconductor industry geopolitical simulations.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import itertools
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class AnalysisType(Enum):
    """Types of scenario analysis."""
    BASELINE_COMPARISON = "baseline_comparison"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    MONTE_CARLO = "monte_carlo"
    STRESS_TESTING = "stress_testing"
    OPTIMIZATION = "optimization"
    CLUSTERING = "clustering"
    TREND_ANALYSIS = "trend_analysis"

class VariableType(Enum):
    """Types of analysis variables."""
    INPUT_PARAMETER = "input_parameter"
    OUTPUT_METRIC = "output_metric"
    INTERMEDIATE_STATE = "intermediate_state"
    DECISION_VARIABLE = "decision_variable"

class RiskLevel(Enum):
    """Risk level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ScenarioDefinition:
    """Definition of a simulation scenario."""
    scenario_id: str
    scenario_name: str
    description: str
    parameter_overrides: Dict[str, Any]
    initial_conditions: Dict[str, Any]
    simulation_length: int  # Years
    monte_carlo_runs: int
    tags: List[str]
    created_date: datetime

@dataclass
class SensitivityParameter:
    """Parameter for sensitivity analysis."""
    parameter_name: str
    parameter_path: str  # Path to parameter in simulation
    base_value: float
    variation_range: Tuple[float, float]  # (min, max)
    distribution_type: str  # "uniform", "normal", "triangular"
    distribution_params: Dict[str, float]
    unit: str
    description: str

@dataclass
class ScenarioResult:
    """Results from a scenario simulation."""
    scenario_id: str
    run_id: str
    execution_time: float
    key_metrics: Dict[str, float]
    time_series_data: Dict[str, List[float]]
    agent_states: Dict[str, Any]
    market_conditions: Dict[str, Any]
    geopolitical_state: Dict[str, Any]
    supply_chain_metrics: Dict[str, Any]
    energy_metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str]

@dataclass
class ComparisonResult:
    """Results from scenario comparison analysis."""
    comparison_id: str
    scenarios_compared: List[str]
    metric_differences: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, Dict[str, float]]
    ranking_by_metric: Dict[str, List[str]]
    risk_assessment: Dict[str, RiskLevel]
    recommendations: List[str]

class ScenarioAnalyzer:
    """
    Comprehensive scenario analysis and comparison system.
    
    Capabilities:
    - Multi-scenario comparison and benchmarking
    - Sensitivity analysis and parameter optimization
    - Monte Carlo simulation and uncertainty quantification
    - Stress testing and resilience analysis
    - Statistical analysis and hypothesis testing
    - Risk assessment and scenario ranking
    - Strategic planning and decision support
    """
    
    def __init__(self):
        # Scenario and analysis data
        self.scenarios: Dict[str, ScenarioDefinition] = {}
        self.scenario_results: Dict[str, List[ScenarioResult]] = {}
        self.sensitivity_parameters: Dict[str, SensitivityParameter] = {}
        
        # Analysis results
        self.comparison_results: Dict[str, ComparisonResult] = {}
        self.sensitivity_results: Dict[str, Any] = {}
        self.optimization_results: Dict[str, Any] = {}
        
        # Analysis configuration
        self.key_metrics = self._define_key_metrics()
        self.analysis_templates = self._define_analysis_templates()
        
        # Initialize with example scenarios
        self._initialize_example_scenarios()
        self._initialize_sensitivity_parameters()
    
    def _define_key_metrics(self) -> Dict[str, Dict[str, str]]:
        """Define key metrics for scenario analysis."""
        
        return {
            "market_performance": {
                "total_market_size": "Total addressable market size",
                "market_growth_rate": "Annual market growth rate",
                "price_volatility": "Price volatility index",
                "supply_demand_balance": "Supply-demand balance ratio"
            },
            "competitive_dynamics": {
                "market_concentration": "Market concentration (HHI)",
                "competitive_intensity": "Competitive intensity score",
                "innovation_rate": "Technology innovation rate",
                "market_share_stability": "Market share stability index"
            },
            "supply_chain_resilience": {
                "supply_chain_risk": "Overall supply chain risk score",
                "disruption_frequency": "Average disruption frequency",
                "recovery_time": "Average recovery time from disruptions",
                "network_redundancy": "Supply chain redundancy score"
            },
            "geopolitical_impact": {
                "trade_restriction_severity": "Trade restriction impact score",
                "alliance_stability": "Alliance stability index",
                "technology_transfer_risk": "Technology transfer risk",
                "regulatory_compliance_cost": "Regulatory compliance cost ratio"
            },
            "environmental_sustainability": {
                "carbon_intensity": "Carbon intensity per unit output",
                "energy_efficiency": "Energy efficiency improvement rate",
                "sustainability_score": "Overall sustainability score",
                "regulatory_risk": "Environmental regulatory risk"
            },
            "financial_performance": {
                "revenue_growth": "Annual revenue growth rate",
                "profit_margin": "Operating profit margin",
                "capex_intensity": "Capital expenditure intensity",
                "return_on_investment": "Return on investment"
            }
        }
    
    def _define_analysis_templates(self) -> Dict[str, Dict[str, Any]]:
        """Define standard analysis templates."""
        
        return {
            "baseline_vs_stress": {
                "description": "Compare baseline scenario against stress test scenarios",
                "scenarios": ["baseline", "trade_war", "supply_disruption", "technology_ban"],
                "focus_metrics": ["supply_chain_risk", "revenue_growth", "market_share_stability"],
                "analysis_type": AnalysisType.BASELINE_COMPARISON
            },
            "technology_leadership_race": {
                "description": "Analyze different technology development strategies",
                "scenarios": ["aggressive_rd", "partnership_focus", "acquisition_strategy"],
                "focus_metrics": ["innovation_rate", "competitive_intensity", "market_concentration"],
                "analysis_type": AnalysisType.BASELINE_COMPARISON
            },
            "geopolitical_tensions": {
                "description": "Assess impact of varying geopolitical tension levels",
                "scenarios": ["cooperation", "competition", "conflict"],
                "focus_metrics": ["trade_restriction_severity", "alliance_stability", "technology_transfer_risk"],
                "analysis_type": AnalysisType.BASELINE_COMPARISON
            },
            "sustainability_pathways": {
                "description": "Compare different sustainability strategy approaches",
                "scenarios": ["carbon_neutral_2030", "renewable_focus", "efficiency_first"],
                "focus_metrics": ["carbon_intensity", "energy_efficiency", "sustainability_score"],
                "analysis_type": AnalysisType.BASELINE_COMPARISON
            }
        }
    
    def _initialize_example_scenarios(self):
        """Initialize example scenarios for analysis."""
        
        # Baseline scenario
        self.scenarios["baseline"] = ScenarioDefinition(
            scenario_id="baseline",
            scenario_name="Baseline Projection",
            description="Current trends continuation with moderate growth",
            parameter_overrides={},
            initial_conditions={
                "market_growth_rate": 0.08,
                "technology_advancement_rate": 0.12,
                "geopolitical_tension": 0.3,
                "supply_chain_stability": 0.8
            },
            simulation_length=10,
            monte_carlo_runs=1000,
            tags=["baseline", "current_trends"],
            created_date=datetime.now()
        )
        
        # Trade war scenario
        self.scenarios["trade_war"] = ScenarioDefinition(
            scenario_id="trade_war",
            scenario_name="Escalating Trade War",
            description="Increasing trade restrictions and technology export controls",
            parameter_overrides={
                "export_control_stringency": 0.8,
                "tariff_rates": {"china": 0.25, "others": 0.05}
            },
            initial_conditions={
                "market_growth_rate": 0.05,
                "technology_advancement_rate": 0.08,
                "geopolitical_tension": 0.8,
                "supply_chain_stability": 0.5
            },
            simulation_length=10,
            monte_carlo_runs=1000,
            tags=["stress_test", "geopolitical", "trade"],
            created_date=datetime.now()
        )
        
        # Technology cooperation scenario
        self.scenarios["tech_cooperation"] = ScenarioDefinition(
            scenario_id="tech_cooperation",
            scenario_name="Enhanced Technology Cooperation",
            description="Increased international cooperation in semiconductor technology",
            parameter_overrides={
                "technology_sharing_rate": 0.4,
                "joint_rd_investment": 0.3
            },
            initial_conditions={
                "market_growth_rate": 0.12,
                "technology_advancement_rate": 0.18,
                "geopolitical_tension": 0.1,
                "supply_chain_stability": 0.9
            },
            simulation_length=10,
            monte_carlo_runs=1000,
            tags=["optimistic", "cooperation", "technology"],
            created_date=datetime.now()
        )
        
        # Supply chain disruption scenario
        self.scenarios["supply_disruption"] = ScenarioDefinition(
            scenario_id="supply_disruption",
            scenario_name="Major Supply Chain Disruption",
            description="Significant supply chain disruptions from natural disasters or conflicts",
            parameter_overrides={
                "disruption_frequency": 0.3,
                "disruption_severity": 0.7
            },
            initial_conditions={
                "market_growth_rate": 0.03,
                "technology_advancement_rate": 0.10,
                "geopolitical_tension": 0.6,
                "supply_chain_stability": 0.3
            },
            simulation_length=10,
            monte_carlo_runs=1000,
            tags=["stress_test", "supply_chain", "disruption"],
            created_date=datetime.now()
        )
    
    def _initialize_sensitivity_parameters(self):
        """Initialize parameters for sensitivity analysis."""
        
        self.sensitivity_parameters["market_growth_rate"] = SensitivityParameter(
            parameter_name="market_growth_rate",
            parameter_path="market.growth_rate",
            base_value=0.08,
            variation_range=(0.02, 0.15),
            distribution_type="normal",
            distribution_params={"mean": 0.08, "std": 0.02},
            unit="percentage",
            description="Annual semiconductor market growth rate"
        )
        
        self.sensitivity_parameters["geopolitical_tension"] = SensitivityParameter(
            parameter_name="geopolitical_tension",
            parameter_path="geopolitical.tension_level",
            base_value=0.3,
            variation_range=(0.0, 1.0),
            distribution_type="uniform",
            distribution_params={"low": 0.0, "high": 1.0},
            unit="index (0-1)",
            description="Overall geopolitical tension level"
        )
        
        self.sensitivity_parameters["technology_advancement_rate"] = SensitivityParameter(
            parameter_name="technology_advancement_rate",
            parameter_path="technology.advancement_rate",
            base_value=0.12,
            variation_range=(0.05, 0.20),
            distribution_type="triangular",
            distribution_params={"low": 0.05, "mode": 0.12, "high": 0.20},
            unit="percentage",
            description="Rate of semiconductor technology advancement"
        )
        
        self.sensitivity_parameters["supply_chain_stability"] = SensitivityParameter(
            parameter_name="supply_chain_stability",
            parameter_path="supply_chain.stability_index",
            base_value=0.8,
            variation_range=(0.3, 0.95),
            distribution_type="normal",
            distribution_params={"mean": 0.8, "std": 0.1},
            unit="index (0-1)",
            description="Overall supply chain stability index"
        )
        
        self.sensitivity_parameters["energy_costs"] = SensitivityParameter(
            parameter_name="energy_costs",
            parameter_path="energy.cost_per_kwh",
            base_value=0.12,
            variation_range=(0.08, 0.25),
            distribution_type="normal",
            distribution_params={"mean": 0.12, "std": 0.03},
            unit="$/kWh",
            description="Industrial electricity cost per kWh"
        )
    
    def run_scenario_comparison(self, scenario_ids: List[str], 
                              focus_metrics: Optional[List[str]] = None,
                              confidence_level: float = 0.95) -> ComparisonResult:
        """Run comprehensive scenario comparison analysis."""
        
        if not all(sid in self.scenarios for sid in scenario_ids):
            missing = [sid for sid in scenario_ids if sid not in self.scenarios]
            raise ValueError(f"Scenarios not found: {missing}")
        
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate scenarios if results don't exist
        for scenario_id in scenario_ids:
            if scenario_id not in self.scenario_results:
                self._simulate_scenario(scenario_id)
        
        # Extract and compare metrics
        scenario_metrics = {}
        for scenario_id in scenario_ids:
            scenario_metrics[scenario_id] = self._extract_scenario_metrics(scenario_id)
        
        # Calculate metric differences
        metric_differences = self._calculate_metric_differences(scenario_metrics)
        
        # Statistical significance testing
        statistical_significance = self._test_statistical_significance(
            scenario_metrics, confidence_level)
        
        # Rank scenarios by metrics
        ranking_by_metric = self._rank_scenarios_by_metrics(scenario_metrics, focus_metrics)
        
        # Risk assessment
        risk_assessment = self._assess_scenario_risks(scenario_metrics)
        
        # Generate recommendations
        recommendations = self._generate_comparison_recommendations(
            scenario_metrics, metric_differences, statistical_significance)
        
        result = ComparisonResult(
            comparison_id=comparison_id,
            scenarios_compared=scenario_ids,
            metric_differences=metric_differences,
            statistical_significance=statistical_significance,
            ranking_by_metric=ranking_by_metric,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
        
        self.comparison_results[comparison_id] = result
        return result
    
    def run_sensitivity_analysis(self, scenario_id: str, 
                               parameters: Optional[List[str]] = None,
                               sample_size: int = 10000) -> Dict[str, Any]:
        """Run comprehensive sensitivity analysis."""
        
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        # Select parameters for analysis
        if parameters is None:
            parameters = list(self.sensitivity_parameters.keys())
        
        # Generate parameter samples
        parameter_samples = self._generate_parameter_samples(parameters, sample_size)
        
        # Run simulations
        simulation_results = []
        for i, sample in enumerate(parameter_samples):
            result = self._simulate_with_parameters(scenario_id, sample)
            simulation_results.append(result)
            
            if i % 1000 == 0:  # Progress update
                print(f"Sensitivity analysis progress: {i}/{sample_size}")
        
        # Analyze sensitivity
        sensitivity_indices = self._calculate_sensitivity_indices(
            parameter_samples, simulation_results, parameters)
        
        # Statistical analysis
        correlation_analysis = self._perform_correlation_analysis(
            parameter_samples, simulation_results)
        
        # Interaction effects
        interaction_effects = self._analyze_interaction_effects(
            parameter_samples, simulation_results, parameters)
        
        # Threshold analysis
        threshold_analysis = self._perform_threshold_analysis(
            parameter_samples, simulation_results)
        
        result = {
            "scenario_id": scenario_id,
            "analysis_date": datetime.now(),
            "sample_size": sample_size,
            "parameters_analyzed": parameters,
            "sensitivity_indices": sensitivity_indices,
            "correlation_analysis": correlation_analysis,
            "interaction_effects": interaction_effects,
            "threshold_analysis": threshold_analysis,
            "key_insights": self._generate_sensitivity_insights(sensitivity_indices, correlation_analysis)
        }
        
        self.sensitivity_results[f"{scenario_id}_sensitivity"] = result
        return result
    
    def _simulate_scenario(self, scenario_id: str) -> List[ScenarioResult]:
        """Simulate a scenario (placeholder for actual simulation)."""
        
        scenario = self.scenarios[scenario_id]
        results = []
        
        # Placeholder simulation - would integrate with actual simulation engine
        for run in range(min(100, scenario.monte_carlo_runs)):  # Limited for example
            # Generate synthetic results based on scenario parameters
            result = ScenarioResult(
                scenario_id=scenario_id,
                run_id=f"{scenario_id}_run_{run}",
                execution_time=np.random.uniform(10, 60),  # Simulation time in seconds
                key_metrics=self._generate_synthetic_metrics(scenario),
                time_series_data=self._generate_synthetic_time_series(scenario),
                agent_states={},
                market_conditions={},
                geopolitical_state={},
                supply_chain_metrics={},
                energy_metrics={},
                success=True,
                error_message=None
            )
            results.append(result)
        
        self.scenario_results[scenario_id] = results
        return results
    
    def _generate_synthetic_metrics(self, scenario: ScenarioDefinition) -> Dict[str, float]:
        """Generate synthetic metrics based on scenario parameters."""
        
        # Base metrics modified by scenario conditions
        base_metrics = {
            "total_market_size": 550.0,  # Billion USD
            "market_growth_rate": scenario.initial_conditions.get("market_growth_rate", 0.08),
            "supply_chain_risk": 1.0 - scenario.initial_conditions.get("supply_chain_stability", 0.8),
            "competitive_intensity": 0.7,
            "innovation_rate": scenario.initial_conditions.get("technology_advancement_rate", 0.12),
            "carbon_intensity": 3.2,
            "revenue_growth": scenario.initial_conditions.get("market_growth_rate", 0.08) * 0.9,
            "profit_margin": 0.25
        }
        
        # Add noise and scenario-specific adjustments
        for metric, value in base_metrics.items():
            # Add random variation
            noise = np.random.normal(0, 0.05 * value)
            base_metrics[metric] = max(0, value + noise)
        
        return base_metrics
    
    def _generate_synthetic_time_series(self, scenario: ScenarioDefinition) -> Dict[str, List[float]]:
        """Generate synthetic time series data."""
        
        time_steps = scenario.simulation_length * 4  # Quarterly data
        
        return {
            "market_size": np.cumsum(np.random.normal(2, 0.5, time_steps)).tolist(),
            "supply_chain_disruptions": np.random.poisson(0.5, time_steps).tolist(),
            "geopolitical_tension": np.clip(
                np.random.normal(scenario.initial_conditions.get("geopolitical_tension", 0.3), 
                               0.1, time_steps), 0, 1).tolist()
        }
    
    def _extract_scenario_metrics(self, scenario_id: str) -> Dict[str, List[float]]:
        """Extract metrics from scenario simulation results."""
        
        results = self.scenario_results[scenario_id]
        metrics = {}
        
        # Aggregate metrics across all runs
        for metric_name in results[0].key_metrics.keys():
            metric_values = [result.key_metrics[metric_name] for result in results]
            metrics[metric_name] = metric_values
        
        return metrics
    
    def _calculate_metric_differences(self, scenario_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
        """Calculate differences between scenario metrics."""
        
        differences = {}
        scenario_ids = list(scenario_metrics.keys())
        
        for i, scenario1 in enumerate(scenario_ids):
            for j, scenario2 in enumerate(scenario_ids[i+1:], i+1):
                comparison_key = f"{scenario1}_vs_{scenario2}"
                differences[comparison_key] = {}
                
                for metric_name in scenario_metrics[scenario1].keys():
                    values1 = scenario_metrics[scenario1][metric_name]
                    values2 = scenario_metrics[scenario2][metric_name]
                    
                    mean_diff = np.mean(values1) - np.mean(values2)
                    pct_diff = (mean_diff / np.mean(values2)) * 100 if np.mean(values2) != 0 else 0
                    
                    differences[comparison_key][metric_name] = {
                        "absolute_difference": mean_diff,
                        "percentage_difference": pct_diff,
                        "std_dev_ratio": np.std(values1) / np.std(values2) if np.std(values2) != 0 else 1
                    }
        
        return differences
    
    def _test_statistical_significance(self, scenario_metrics: Dict[str, Dict[str, List[float]]], 
                                     confidence_level: float) -> Dict[str, Dict[str, float]]:
        """Test statistical significance of differences between scenarios."""
        
        significance_results = {}
        scenario_ids = list(scenario_metrics.keys())
        
        for i, scenario1 in enumerate(scenario_ids):
            for j, scenario2 in enumerate(scenario_ids[i+1:], i+1):
                comparison_key = f"{scenario1}_vs_{scenario2}"
                significance_results[comparison_key] = {}
                
                for metric_name in scenario_metrics[scenario1].keys():
                    values1 = scenario_metrics[scenario1][metric_name]
                    values2 = scenario_metrics[scenario2][metric_name]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(values1, values2)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
                    effect_size = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std != 0 else 0
                    
                    significance_results[comparison_key][metric_name] = {
                        "p_value": p_value,
                        "is_significant": p_value < (1 - confidence_level),
                        "effect_size": effect_size,
                        "confidence_interval": stats.t.interval(confidence_level, 
                                                               len(values1) + len(values2) - 2,
                                                               np.mean(values1) - np.mean(values2),
                                                               stats.sem(values1 + values2))
                    }
        
        return significance_results
    
    def _rank_scenarios_by_metrics(self, scenario_metrics: Dict[str, Dict[str, List[float]]], 
                                 focus_metrics: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """Rank scenarios by different metrics."""
        
        rankings = {}
        metrics_to_rank = focus_metrics if focus_metrics else list(next(iter(scenario_metrics.values())).keys())
        
        for metric_name in metrics_to_rank:
            scenario_means = {}
            for scenario_id, metrics in scenario_metrics.items():
                if metric_name in metrics:
                    scenario_means[scenario_id] = np.mean(metrics[metric_name])
            
            # Sort scenarios by metric value (higher is better for most metrics)
            # For risk metrics, lower is better
            reverse_sort = "risk" not in metric_name.lower()
            sorted_scenarios = sorted(scenario_means.items(), 
                                    key=lambda x: x[1], reverse=reverse_sort)
            
            rankings[metric_name] = [scenario_id for scenario_id, _ in sorted_scenarios]
        
        return rankings
    
    def _assess_scenario_risks(self, scenario_metrics: Dict[str, Dict[str, List[float]]]) -> Dict[str, RiskLevel]:
        """Assess risk levels for each scenario."""
        
        risk_assessment = {}
        
        for scenario_id, metrics in scenario_metrics.items():
            risk_factors = []
            
            # Analyze risk indicators
            if "supply_chain_risk" in metrics:
                risk_factors.append(np.mean(metrics["supply_chain_risk"]))
            
            if "market_growth_rate" in metrics:
                growth_volatility = np.std(metrics["market_growth_rate"])
                risk_factors.append(growth_volatility)
            
            if "competitive_intensity" in metrics:
                risk_factors.append(np.mean(metrics["competitive_intensity"]))
            
            # Calculate overall risk score
            if risk_factors:
                avg_risk = np.mean(risk_factors)
                if avg_risk < 0.3:
                    risk_level = RiskLevel.LOW
                elif avg_risk < 0.6:
                    risk_level = RiskLevel.MEDIUM
                elif avg_risk < 0.8:
                    risk_level = RiskLevel.HIGH
                else:
                    risk_level = RiskLevel.CRITICAL
            else:
                risk_level = RiskLevel.MEDIUM
            
            risk_assessment[scenario_id] = risk_level
        
        return risk_assessment
    
    def _generate_comparison_recommendations(self, scenario_metrics: Dict, 
                                           metric_differences: Dict, 
                                           statistical_significance: Dict) -> List[str]:
        """Generate strategic recommendations based on scenario comparison."""
        
        recommendations = []
        
        # Find best performing scenarios
        scenario_scores = {}
        for scenario_id in scenario_metrics.keys():
            score = 0
            for metric_name, values in scenario_metrics[scenario_id].items():
                if "growth" in metric_name or "efficiency" in metric_name:
                    score += np.mean(values)
                elif "risk" in metric_name:
                    score -= np.mean(values)
            scenario_scores[scenario_id] = score
        
        best_scenario = max(scenario_scores.items(), key=lambda x: x[1])[0]
        worst_scenario = min(scenario_scores.items(), key=lambda x: x[1])[0]
        
        recommendations.append(f"Scenario '{best_scenario}' shows strongest overall performance")
        recommendations.append(f"Scenario '{worst_scenario}' carries highest risk profile")
        
        # Identify key differentiating factors
        for comparison_key, differences in metric_differences.items():
            for metric_name, diff_data in differences.items():
                if abs(diff_data["percentage_difference"]) > 20:  # Significant difference
                    recommendations.append(
                        f"{metric_name} shows {abs(diff_data['percentage_difference']):.1f}% "
                        f"difference between scenarios in {comparison_key}"
                    )
        
        # Strategic recommendations
        recommendations.extend([
            "Monitor key risk indicators identified in scenario analysis",
            "Develop contingency plans for high-risk scenarios", 
            "Invest in capabilities that perform well across multiple scenarios",
            "Regular scenario review and updating recommended"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_parameter_samples(self, parameters: List[str], sample_size: int) -> List[Dict[str, float]]:
        """Generate parameter samples for sensitivity analysis."""
        
        samples = []
        
        for _ in range(sample_size):
            sample = {}
            for param_name in parameters:
                param_def = self.sensitivity_parameters[param_name]
                
                if param_def.distribution_type == "uniform":
                    value = np.random.uniform(param_def.variation_range[0], param_def.variation_range[1])
                elif param_def.distribution_type == "normal":
                    value = np.random.normal(param_def.distribution_params["mean"], 
                                           param_def.distribution_params["std"])
                    value = np.clip(value, param_def.variation_range[0], param_def.variation_range[1])
                elif param_def.distribution_type == "triangular":
                    value = np.random.triangular(param_def.distribution_params["low"],
                                                param_def.distribution_params["mode"],
                                                param_def.distribution_params["high"])
                else:
                    value = param_def.base_value
                
                sample[param_name] = value
            
            samples.append(sample)
        
        return samples
    
    def _simulate_with_parameters(self, scenario_id: str, parameters: Dict[str, float]) -> Dict[str, float]:
        """Simulate scenario with specific parameter values."""
        
        # Placeholder for actual simulation with parameter overrides
        base_scenario = self.scenarios[scenario_id]
        
        # Modify base metrics based on parameters
        modified_metrics = {
            "total_market_size": 550.0 * (1 + parameters.get("market_growth_rate", 0.08)),
            "supply_chain_risk": 1.0 - parameters.get("supply_chain_stability", 0.8),
            "innovation_rate": parameters.get("technology_advancement_rate", 0.12),
            "competitive_intensity": 0.7 * (1 + parameters.get("geopolitical_tension", 0.3)),
            "energy_efficiency": 0.8 / (1 + parameters.get("energy_costs", 0.12) * 5),
            "revenue_growth": parameters.get("market_growth_rate", 0.08) * 0.9
        }
        
        # Add parameter interactions
        if parameters.get("geopolitical_tension", 0) > 0.6:
            modified_metrics["supply_chain_risk"] *= 1.5
            modified_metrics["innovation_rate"] *= 0.8
        
        return modified_metrics
    
    def _calculate_sensitivity_indices(self, parameter_samples: List[Dict[str, float]], 
                                     simulation_results: List[Dict[str, float]], 
                                     parameters: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate sensitivity indices for parameters."""
        
        sensitivity_indices = {}
        
        # Convert to arrays for analysis
        param_array = np.array([[sample[param] for param in parameters] for sample in parameter_samples])
        
        for metric_name in simulation_results[0].keys():
            metric_values = np.array([result[metric_name] for result in simulation_results])
            sensitivity_indices[metric_name] = {}
            
            # Calculate first-order sensitivity indices
            for i, param_name in enumerate(parameters):
                param_values = param_array[:, i]
                
                # Correlation-based sensitivity
                correlation = np.corrcoef(param_values, metric_values)[0, 1]
                sensitivity_indices[metric_name][param_name] = {
                    "correlation": correlation,
                    "correlation_squared": correlation ** 2,
                    "rank_correlation": stats.spearmanr(param_values, metric_values)[0]
                }
        
        return sensitivity_indices
    
    def _perform_correlation_analysis(self, parameter_samples: List[Dict[str, float]], 
                                    simulation_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform detailed correlation analysis."""
        
        # Create correlation matrix
        all_data = []
        column_names = []
        
        # Add parameters
        for param_name in parameter_samples[0].keys():
            all_data.append([sample[param_name] for sample in parameter_samples])
            column_names.append(f"param_{param_name}")
        
        # Add metrics
        for metric_name in simulation_results[0].keys():
            all_data.append([result[metric_name] for result in simulation_results])
            column_names.append(f"metric_{metric_name}")
        
        data_array = np.array(all_data).T
        correlation_matrix = np.corrcoef(data_array.T)
        
        return {
            "correlation_matrix": correlation_matrix.tolist(),
            "column_names": column_names,
            "strong_correlations": self._identify_strong_correlations(correlation_matrix, column_names),
            "parameter_metric_correlations": self._extract_parameter_metric_correlations(
                correlation_matrix, column_names)
        }
    
    def _identify_strong_correlations(self, correlation_matrix: np.ndarray, 
                                    column_names: List[str], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify strong correlations in the analysis."""
        
        strong_correlations = []
        n = len(column_names)
        
        for i in range(n):
            for j in range(i+1, n):
                correlation = correlation_matrix[i, j]
                if abs(correlation) >= threshold:
                    strong_correlations.append({
                        "variable1": column_names[i],
                        "variable2": column_names[j],
                        "correlation": correlation,
                        "strength": "Very Strong" if abs(correlation) >= 0.9 else "Strong"
                    })
        
        return sorted(strong_correlations, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def _extract_parameter_metric_correlations(self, correlation_matrix: np.ndarray, 
                                             column_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Extract correlations between parameters and metrics."""
        
        param_indices = [i for i, name in enumerate(column_names) if name.startswith("param_")]
        metric_indices = [i for i, name in enumerate(column_names) if name.startswith("metric_")]
        
        correlations = {}
        
        for param_idx in param_indices:
            param_name = column_names[param_idx].replace("param_", "")
            correlations[param_name] = {}
            
            for metric_idx in metric_indices:
                metric_name = column_names[metric_idx].replace("metric_", "")
                correlations[param_name][metric_name] = correlation_matrix[param_idx, metric_idx]
        
        return correlations
    
    def _analyze_interaction_effects(self, parameter_samples: List[Dict[str, float]], 
                                   simulation_results: List[Dict[str, float]], 
                                   parameters: List[str]) -> Dict[str, Any]:
        """Analyze parameter interaction effects."""
        
        interaction_effects = {}
        
        # Analyze pairwise interactions
        for i, param1 in enumerate(parameters):
            for j, param2 in enumerate(parameters[i+1:], i+1):
                interaction_key = f"{param1}_x_{param2}"
                interaction_effects[interaction_key] = {}
                
                # Create interaction term
                interaction_values = [
                    sample[param1] * sample[param2] for sample in parameter_samples
                ]
                
                # Analyze effect on each metric
                for metric_name in simulation_results[0].keys():
                    metric_values = [result[metric_name] for result in simulation_results]
                    
                    correlation = np.corrcoef(interaction_values, metric_values)[0, 1]
                    interaction_effects[interaction_key][metric_name] = correlation
        
        return interaction_effects
    
    def _perform_threshold_analysis(self, parameter_samples: List[Dict[str, float]], 
                                  simulation_results: List[Dict[str, float]]) -> Dict[str, Any]:
        """Perform threshold analysis to identify critical parameter values."""
        
        threshold_analysis = {}
        
        for metric_name in simulation_results[0].keys():
            metric_values = np.array([result[metric_name] for result in simulation_results])
            
            # Define thresholds (e.g., 10th and 90th percentiles)
            low_threshold = np.percentile(metric_values, 10)
            high_threshold = np.percentile(metric_values, 90)
            
            threshold_analysis[metric_name] = {
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "critical_parameters": self._identify_critical_parameters_for_metric(
                    parameter_samples, metric_values, low_threshold, high_threshold)
            }
        
        return threshold_analysis
    
    def _identify_critical_parameters_for_metric(self, parameter_samples: List[Dict[str, float]], 
                                               metric_values: np.ndarray, 
                                               low_threshold: float, 
                                               high_threshold: float) -> Dict[str, Any]:
        """Identify parameters most associated with extreme metric values."""
        
        critical_params = {}
        
        # Find samples with extreme values
        low_indices = np.where(metric_values <= low_threshold)[0]
        high_indices = np.where(metric_values >= high_threshold)[0]
        
        if len(low_indices) > 0 and len(high_indices) > 0:
            for param_name in parameter_samples[0].keys():
                low_param_values = [parameter_samples[i][param_name] for i in low_indices]
                high_param_values = [parameter_samples[i][param_name] for i in high_indices]
                
                # Test for significant difference
                t_stat, p_value = stats.ttest_ind(low_param_values, high_param_values)
                
                critical_params[param_name] = {
                    "low_mean": np.mean(low_param_values),
                    "high_mean": np.mean(high_param_values),
                    "difference": np.mean(high_param_values) - np.mean(low_param_values),
                    "p_value": p_value,
                    "is_critical": p_value < 0.05
                }
        
        return critical_params
    
    def _generate_sensitivity_insights(self, sensitivity_indices: Dict, 
                                     correlation_analysis: Dict) -> List[str]:
        """Generate key insights from sensitivity analysis."""
        
        insights = []
        
        # Find most influential parameters
        param_influence = {}
        for metric_name, param_sensitivities in sensitivity_indices.items():
            for param_name, sensitivity_data in param_sensitivities.items():
                if param_name not in param_influence:
                    param_influence[param_name] = []
                param_influence[param_name].append(abs(sensitivity_data["correlation"]))
        
        # Calculate average influence
        avg_influence = {param: np.mean(influences) for param, influences in param_influence.items()}
        most_influential = max(avg_influence.items(), key=lambda x: x[1])
        least_influential = min(avg_influence.items(), key=lambda x: x[1])
        
        insights.append(f"Most influential parameter: {most_influential[0]} "
                       f"(average correlation: {most_influential[1]:.3f})")
        insights.append(f"Least influential parameter: {least_influential[0]} "
                       f"(average correlation: {least_influential[1]:.3f})")
        
        # Identify strong relationships
        strong_correlations = correlation_analysis.get("strong_correlations", [])
        if strong_correlations:
            top_correlation = strong_correlations[0]
            insights.append(f"Strongest relationship: {top_correlation['variable1']} - "
                          f"{top_correlation['variable2']} (correlation: {top_correlation['correlation']:.3f})")
        
        return insights
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all scenario analyses."""
        
        return {
            "scenarios_defined": len(self.scenarios),
            "scenario_results_available": len(self.scenario_results),
            "comparisons_completed": len(self.comparison_results),
            "sensitivity_analyses": len(self.sensitivity_results),
            "key_metrics_tracked": len(self.key_metrics),
            "analysis_templates": list(self.analysis_templates.keys()),
            "sensitivity_parameters": list(self.sensitivity_parameters.keys()),
            "capabilities": [
                "Multi-scenario comparison and benchmarking",
                "Comprehensive sensitivity analysis",
                "Statistical significance testing",
                "Risk assessment and scenario ranking",
                "Parameter optimization and threshold analysis",
                "Correlation and interaction analysis",
                "Strategic recommendation generation"
            ]
        }