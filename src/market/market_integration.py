"""
Market Dynamics Engine Integration

Integrates demand forecasting, supply capacity, and pricing models into a unified
market dynamics simulation engine for comprehensive semiconductor market modeling.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

# Add project root to path for imports
sys.path.append('src')

from config.constants import *
from market.demand_models import DemandForecastModel, MarketSegmentDemand
from market.supply_models import SupplyCapacityModel, CapacityConstraints
from market.pricing_models import PricingMechanismModel, MarketPricing

class MarketScenario(Enum):
    """Market scenario types for simulation."""
    BASELINE = "baseline"
    ECONOMIC_RECESSION = "economic_recession" 
    TRADE_WAR_ESCALATION = "trade_war_escalation"
    AI_BOOM = "ai_boom"
    SUPPLY_CHAIN_CRISIS = "supply_chain_crisis"
    TECHNOLOGY_BREAKTHROUGH = "technology_breakthrough"
    GEOPOLITICAL_CRISIS = "geopolitical_crisis"

@dataclass
class MarketState:
    """Current market state snapshot."""
    timestamp: str
    total_demand: Dict[str, float]  # By process node
    total_supply: Dict[str, float]  # By process node
    market_prices: Dict[str, float]  # By product category
    utilization_rates: Dict[str, float]  # By process node
    market_conditions: Dict[str, str]  # By process node
    supply_demand_ratios: Dict[str, float]

@dataclass
class MarketSimulationResults:
    """Complete market simulation results."""
    scenario: str
    time_horizon_quarters: int
    quarterly_market_states: List[MarketState]
    demand_evolution: Dict[str, List[float]]
    supply_evolution: Dict[str, List[float]]
    price_evolution: Dict[str, List[float]]
    market_value_evolution: List[float]
    key_metrics: Dict[str, Any]
    risk_analysis: Dict[str, Any]

class MarketDynamicsEngine:
    """
    Integrated market dynamics simulation engine.
    
    Coordinates:
    - Demand forecasting across market segments
    - Supply capacity modeling and constraints
    - Dynamic pricing mechanisms
    - Market equilibrium calculation
    - Scenario analysis and stress testing
    - Risk assessment and bottleneck identification
    """
    
    def __init__(self, forecast_horizon_years: int = 10):
        self.forecast_horizon = forecast_horizon_years
        self.quarters = forecast_horizon_years * 4
        
        # Initialize component models
        self.demand_model = DemandForecastModel(forecast_horizon_years)
        self.supply_model = SupplyCapacityModel()
        self.pricing_model = PricingMechanismModel()
        
        # Market state tracking
        self.current_market_state: Optional[MarketState] = None
        self.market_history: List[MarketState] = []
        
        # Scenario configurations
        self.scenario_configs = self._initialize_scenario_configs()
        
        # Market integration parameters
        self.price_adjustment_speed = 0.3  # How quickly prices adjust to imbalances
        self.capacity_response_lag = 8  # Quarters for capacity to respond to demand
        self.demand_price_feedback = 0.5  # How much demand responds to price changes
        
    def _initialize_scenario_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize scenario configuration parameters."""
        return {
            MarketScenario.BASELINE.value: {
                'demand_modifiers': {'ai_acceleration': 1.0, 'economic_growth': 1.0},
                'supply_modifiers': {'capacity_growth': 1.0, 'utilization_target': 0.85},
                'constraint_modifiers': {'severity_multiplier': 1.0},
                'price_modifiers': {'volatility': 1.0, 'premium': 1.0}
            },
            
            MarketScenario.ECONOMIC_RECESSION.value: {
                'demand_modifiers': {'ai_acceleration': 0.7, 'economic_growth': 0.6},
                'supply_modifiers': {'capacity_growth': 0.5, 'utilization_target': 0.70},
                'constraint_modifiers': {'severity_multiplier': 1.2},
                'price_modifiers': {'volatility': 1.5, 'premium': 0.8}
            },
            
            MarketScenario.TRADE_WAR_ESCALATION.value: {
                'demand_modifiers': {'ai_acceleration': 0.8, 'economic_growth': 0.8},
                'supply_modifiers': {'capacity_growth': 0.6, 'utilization_target': 0.75},
                'constraint_modifiers': {'severity_multiplier': 2.0},
                'price_modifiers': {'volatility': 2.0, 'premium': 1.3}
            },
            
            MarketScenario.AI_BOOM.value: {
                'demand_modifiers': {'ai_acceleration': 2.5, 'economic_growth': 1.2},
                'supply_modifiers': {'capacity_growth': 1.3, 'utilization_target': 0.95},
                'constraint_modifiers': {'severity_multiplier': 1.5},
                'price_modifiers': {'volatility': 1.3, 'premium': 1.4}
            },
            
            MarketScenario.SUPPLY_CHAIN_CRISIS.value: {
                'demand_modifiers': {'ai_acceleration': 0.9, 'economic_growth': 0.85},
                'supply_modifiers': {'capacity_growth': 0.3, 'utilization_target': 0.60},
                'constraint_modifiers': {'severity_multiplier': 3.0},
                'price_modifiers': {'volatility': 2.5, 'premium': 1.8}
            },
            
            MarketScenario.TECHNOLOGY_BREAKTHROUGH.value: {
                'demand_modifiers': {'ai_acceleration': 1.8, 'economic_growth': 1.3},
                'supply_modifiers': {'capacity_growth': 2.0, 'utilization_target': 0.90},
                'constraint_modifiers': {'severity_multiplier': 0.7},
                'price_modifiers': {'volatility': 1.1, 'premium': 0.9}
            },
            
            MarketScenario.GEOPOLITICAL_CRISIS.value: {
                'demand_modifiers': {'ai_acceleration': 0.6, 'economic_growth': 0.7},
                'supply_modifiers': {'capacity_growth': 0.4, 'utilization_target': 0.65},
                'constraint_modifiers': {'severity_multiplier': 2.5},
                'price_modifiers': {'volatility': 3.0, 'premium': 1.6}
            }
        }
    
    def run_market_simulation(self, scenario: str = "baseline", 
                            custom_parameters: Optional[Dict[str, Any]] = None) -> MarketSimulationResults:
        """Run comprehensive market simulation for specified scenario."""
        
        # Apply scenario configuration
        self._configure_scenario(scenario, custom_parameters)
        
        # Generate base demand forecast
        demand_forecasts = self.demand_model.generate_base_demand_forecast()
        
        # Apply scenario modifiers to demand
        demand_forecasts = self.demand_model.apply_scenario_modifiers(scenario, demand_forecasts)
        
        # Initialize simulation tracking
        quarterly_states = []
        demand_evolution = {}
        supply_evolution = {}
        price_evolution = {}
        market_value_evolution = []
        
        # Process node mapping for supply-demand matching
        node_mapping = self._create_node_mapping(demand_forecasts)
        
        # Run quarterly simulation
        for quarter in range(self.quarters):
            # Calculate quarterly demand by process node
            quarterly_demand = self._calculate_quarterly_demand(demand_forecasts, quarter, node_mapping)
            
            # Calculate effective supply capacity
            quarterly_supply = self._calculate_quarterly_supply(quarter)
            
            # Update market conditions based on supply-demand balance
            self._update_market_conditions(quarterly_demand, quarterly_supply, quarter)
            
            # Calculate dynamic pricing
            quarterly_prices = self._calculate_quarterly_pricing(quarterly_demand, quarterly_supply, quarter)
            
            # Apply demand-price feedback
            adjusted_demand = self._apply_price_feedback(quarterly_demand, quarterly_prices, quarter)
            
            # Calculate market value
            quarterly_market_value = self._calculate_market_value(adjusted_demand, quarterly_prices)
            
            # Create market state snapshot
            market_state = MarketState(
                timestamp=f"Q{quarter + 1}",
                total_demand=adjusted_demand,
                total_supply=quarterly_supply,
                market_prices=quarterly_prices,
                utilization_rates=self._calculate_utilization_rates(adjusted_demand, quarterly_supply),
                market_conditions={node: self.pricing_model.current_market_conditions.get(node, "balanced") 
                                 for node in quarterly_demand.keys()},
                supply_demand_ratios={node: quarterly_supply.get(node, 0) / max(adjusted_demand.get(node, 1), 1)
                                    for node in quarterly_demand.keys()}
            )
            
            quarterly_states.append(market_state)
            
            # Track evolution
            for node in quarterly_demand.keys():
                if node not in demand_evolution:
                    demand_evolution[node] = []
                    supply_evolution[node] = []
                    price_evolution[node] = []
                
                demand_evolution[node].append(adjusted_demand.get(node, 0))
                supply_evolution[node].append(quarterly_supply.get(node, 0))
                price_evolution[node].append(quarterly_prices.get(node, 0))
            
            market_value_evolution.append(quarterly_market_value)
            
            # Update models for next quarter
            self._update_models_state(market_state, quarter)
        
        # Calculate key metrics and risk analysis
        key_metrics = self._calculate_key_metrics(quarterly_states, market_value_evolution)
        risk_analysis = self._perform_risk_analysis(quarterly_states)
        
        return MarketSimulationResults(
            scenario=scenario,
            time_horizon_quarters=self.quarters,
            quarterly_market_states=quarterly_states,
            demand_evolution=demand_evolution,
            supply_evolution=supply_evolution,
            price_evolution=price_evolution,
            market_value_evolution=market_value_evolution,
            key_metrics=key_metrics,
            risk_analysis=risk_analysis
        )
    
    def _configure_scenario(self, scenario: str, custom_parameters: Optional[Dict[str, Any]]):
        """Configure models for specific scenario."""
        if scenario not in self.scenario_configs:
            scenario = MarketScenario.BASELINE.value
        
        config = self.scenario_configs[scenario].copy()
        
        # Apply custom parameters if provided
        if custom_parameters:
            for category, params in custom_parameters.items():
                if category in config:
                    config[category].update(params)
        
        # Configure demand model
        demand_config = config['demand_modifiers']
        self.demand_model.update_economic_conditions(
            demand_config.get('economic_growth', 1.0) - 1.0
        )
        
        # Configure supply model constraints
        constraint_config = config['constraint_modifiers']
        severity_multiplier = constraint_config.get('severity_multiplier', 1.0)
        
        # Adjust existing constraints
        for constraint in self.supply_model.active_constraints:
            constraint.severity_level = min(1.0, constraint.severity_level * severity_multiplier)
        
        # Configure pricing model volatility
        price_config = config['price_modifiers']
        # This would be implemented in the pricing model if it had volatility parameters
    
    def _create_node_mapping(self, demand_forecasts: Dict) -> Dict[str, str]:
        """Create mapping from demand segments to supply process nodes."""
        mapping = {}
        
        # Map demand segments to process nodes based on node preferences
        for segment, forecast in demand_forecasts.items():
            # Find the process node with highest preference for this segment
            max_preference = 0
            preferred_node = "28nm"  # Default
            
            for node_demand, preference in forecast.node_preference.items():
                if preference > max_preference:
                    max_preference = preference
                    if node_demand.value == "leading_edge":
                        preferred_node = "5nm"
                    elif node_demand.value == "advanced":
                        preferred_node = "7nm"
                    elif node_demand.value == "mainstream":
                        preferred_node = "16nm"
                    else:
                        preferred_node = "28nm"
            
            mapping[segment.value] = preferred_node
        
        return mapping
    
    def _calculate_quarterly_demand(self, demand_forecasts: Dict, quarter: int, 
                                  node_mapping: Dict[str, str]) -> Dict[str, float]:
        """Calculate quarterly demand by process node."""
        node_demand = {}
        
        for segment, forecast in demand_forecasts.items():
            if quarter < len(forecast.quarterly_demand_units):
                segment_demand = forecast.quarterly_demand_units[quarter]
                
                # Map to process nodes based on preferences
                for node_category, preference in forecast.node_preference.items():
                    # Convert node category to actual process node
                    if node_category.value == "leading_edge":
                        process_nodes = ["3nm", "5nm"]
                    elif node_category.value == "advanced":
                        process_nodes = ["7nm", "10nm"]
                    elif node_category.value == "mainstream":
                        process_nodes = ["16nm", "28nm"]
                    else:  # mature
                        process_nodes = ["28nm", "65nm"]
                    
                    # Distribute demand across nodes in category
                    for node in process_nodes:
                        if node not in node_demand:
                            node_demand[node] = 0
                        node_demand[node] += segment_demand * preference / len(process_nodes)
        
        return node_demand
    
    def _calculate_quarterly_supply(self, quarter: int) -> Dict[str, float]:
        """Calculate quarterly supply capacity by process node."""
        quarterly_supply = {}
        
        process_nodes = ["3nm", "5nm", "7nm", "10nm", "16nm", "28nm", "65nm"]
        
        for node in process_nodes:
            effective_capacity = self.supply_model.calculate_effective_capacity(node)
            monthly_capacity = sum(fab['effective_capacity'] for fab in effective_capacity.values())
            quarterly_supply[node] = monthly_capacity * 3  # Convert to quarterly
        
        return quarterly_supply
    
    def _update_market_conditions(self, demand: Dict[str, float], supply: Dict[str, float], quarter: int):
        """Update market conditions based on supply-demand balance."""
        self.pricing_model.update_market_conditions(supply, demand)
    
    def _calculate_quarterly_pricing(self, demand: Dict[str, float], supply: Dict[str, float], 
                                   quarter: int) -> Dict[str, float]:
        """Calculate quarterly pricing by process node."""
        node_prices = {}
        
        for node in demand.keys():
            # Find corresponding product in pricing model
            product_key = f"{node}_logic"  # Default to logic products
            
            if product_key in self.pricing_model.market_pricing:
                # Calculate dynamic pricing based on current conditions
                pricing_result = self.pricing_model.calculate_dynamic_pricing(
                    product_key=product_key,
                    volume=int(demand.get(node, 1000)),  # Use demand as volume proxy
                    custom_factors={
                        'supply_chain_risk': 0.2 if quarter % 12 < 3 else 0.1,  # Seasonal risk
                        'geopolitical_risk': 0.1
                    }
                )
                node_prices[node] = pricing_result['final_price']
            else:
                # Fallback pricing
                base_prices = {"3nm": 15000, "5nm": 9566, "7nm": 8500, "10nm": 6000, 
                              "16nm": 4000, "28nm": 2500, "65nm": 1500}
                node_prices[node] = base_prices.get(node, 3000)
        
        return node_prices
    
    def _apply_price_feedback(self, demand: Dict[str, float], prices: Dict[str, float], 
                            quarter: int) -> Dict[str, float]:
        """Apply price feedback to demand (demand responds to price changes)."""
        adjusted_demand = demand.copy()
        
        for node in demand.keys():
            if node in prices:
                # Simple price elasticity effect
                base_price = {"3nm": 15000, "5nm": 9566, "7nm": 8500, "10nm": 6000,
                             "16nm": 4000, "28nm": 2500, "65nm": 1500}.get(node, 3000)
                
                price_ratio = prices[node] / base_price
                elasticity = -0.5  # Price elasticity of demand
                
                demand_adjustment = (price_ratio - 1) * elasticity * self.demand_price_feedback
                adjusted_demand[node] = max(0, demand[node] * (1 + demand_adjustment))
        
        return adjusted_demand
    
    def _calculate_market_value(self, demand: Dict[str, float], prices: Dict[str, float]) -> float:
        """Calculate total quarterly market value."""
        total_value = 0
        for node in demand.keys():
            if node in prices:
                total_value += demand[node] * prices[node] / 1e9  # Convert to billions
        return total_value
    
    def _calculate_utilization_rates(self, demand: Dict[str, float], supply: Dict[str, float]) -> Dict[str, float]:
        """Calculate capacity utilization rates."""
        utilization = {}
        for node in demand.keys():
            if node in supply and supply[node] > 0:
                utilization[node] = min(1.0, demand[node] / supply[node])
            else:
                utilization[node] = 1.0  # Assume full utilization if no supply data
        return utilization
    
    def _update_models_state(self, market_state: MarketState, quarter: int):
        """Update model states for next quarter."""
        # Update supply model utilization rates
        for node, utilization in market_state.utilization_rates.items():
            # Update fab utilization in supply model (simplified)
            for fab_id, fab_data in self.supply_model.fab_capacity_data.items():
                if node in fab_data.process_nodes:
                    self.supply_model.update_fab_utilization(fab_id, utilization)
                    break
    
    def _calculate_key_metrics(self, quarterly_states: List[MarketState], 
                             market_values: List[float]) -> Dict[str, Any]:
        """Calculate key performance metrics from simulation."""
        
        if not quarterly_states or not market_values:
            return {}
        
        # Market growth metrics
        initial_value = market_values[0]
        final_value = market_values[-1]
        market_cagr = (final_value / initial_value) ** (4 / len(market_values)) - 1 if initial_value > 0 else 0
        
        # Supply-demand balance metrics
        avg_utilization = {}
        max_utilization = {}
        shortage_periods = {}
        
        for node in quarterly_states[0].total_demand.keys():
            utilizations = [state.utilization_rates.get(node, 0) for state in quarterly_states]
            avg_utilization[node] = np.mean(utilizations)
            max_utilization[node] = max(utilizations)
            shortage_periods[node] = sum(1 for u in utilizations if u > 0.95) / len(utilizations)
        
        # Price volatility
        price_volatility = {}
        for node in quarterly_states[0].market_prices.keys():
            prices = [state.market_prices.get(node, 0) for state in quarterly_states]
            if prices and len(prices) > 1:
                price_volatility[node] = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        return {
            'market_value_cagr': market_cagr,
            'initial_market_value_billions': initial_value,
            'final_market_value_billions': final_value,
            'total_cumulative_value_billions': sum(market_values),
            'avg_capacity_utilization': avg_utilization,
            'peak_capacity_utilization': max_utilization,
            'shortage_risk_by_node': shortage_periods,
            'price_volatility_by_node': price_volatility,
            'avg_supply_demand_ratio': np.mean([
                np.mean(list(state.supply_demand_ratios.values())) for state in quarterly_states
            ])
        }
    
    def _perform_risk_analysis(self, quarterly_states: List[MarketState]) -> Dict[str, Any]:
        """Perform comprehensive risk analysis."""
        
        risk_metrics = {}
        
        # Supply concentration risk
        supply_risk = self.supply_model.assess_supply_chain_risks()
        risk_metrics['supply_chain_risk'] = supply_risk['overall_risk_score']
        
        # Market volatility risk
        volatility_scores = []
        for node in quarterly_states[0].total_demand.keys():
            ratios = [state.supply_demand_ratios.get(node, 1.0) for state in quarterly_states]
            volatility = np.std(ratios) if len(ratios) > 1 else 0
            volatility_scores.append(volatility)
        
        risk_metrics['market_volatility_risk'] = np.mean(volatility_scores)
        
        # Shortage risk
        shortage_risk = {}
        for node in quarterly_states[0].total_demand.keys():
            shortage_quarters = sum(1 for state in quarterly_states 
                                  if state.supply_demand_ratios.get(node, 1.0) < 1.0)
            shortage_risk[node] = shortage_quarters / len(quarterly_states)
        
        risk_metrics['shortage_risk_by_node'] = shortage_risk
        risk_metrics['overall_shortage_risk'] = np.mean(list(shortage_risk.values()))
        
        # Price shock risk
        price_shock_risk = {}
        for node in quarterly_states[0].market_prices.keys():
            prices = [state.market_prices.get(node, 0) for state in quarterly_states]
            if len(prices) > 1:
                max_change = max(abs(prices[i+1] - prices[i]) / prices[i] 
                               for i in range(len(prices)-1) if prices[i] > 0)
                price_shock_risk[node] = max_change
        
        risk_metrics['price_shock_risk'] = price_shock_risk
        
        # Overall risk score
        risk_components = [
            risk_metrics['supply_chain_risk'],
            risk_metrics['market_volatility_risk'],
            risk_metrics['overall_shortage_risk'],
            np.mean(list(price_shock_risk.values())) if price_shock_risk else 0
        ]
        
        risk_metrics['overall_risk_score'] = np.mean(risk_components)
        
        return risk_metrics
    
    def compare_scenarios(self, scenarios: List[str], 
                         custom_parameters: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Compare multiple scenarios and provide comparative analysis."""
        scenario_results = {}
        
        for scenario in scenarios:
            custom_params = custom_parameters.get(scenario) if custom_parameters else None
            result = self.run_market_simulation(scenario, custom_params)
            scenario_results[scenario] = result
        
        # Comparative analysis
        comparison = {
            'scenarios': list(scenarios),
            'market_value_comparison': {},
            'risk_comparison': {},
            'key_insights': []
        }
        
        # Compare market values
        for scenario, result in scenario_results.items():
            comparison['market_value_comparison'][scenario] = {
                'final_value': result.key_metrics['final_market_value_billions'],
                'cagr': result.key_metrics['market_value_cagr'],
                'cumulative_value': result.key_metrics['total_cumulative_value_billions']
            }
        
        # Compare risks
        for scenario, result in scenario_results.items():
            comparison['risk_comparison'][scenario] = {
                'overall_risk': result.risk_analysis['overall_risk_score'],
                'shortage_risk': result.risk_analysis['overall_shortage_risk'],
                'supply_chain_risk': result.risk_analysis['supply_chain_risk']
            }
        
        # Generate insights
        best_value_scenario = max(scenario_results.keys(), 
                                key=lambda s: scenario_results[s].key_metrics['final_market_value_billions'])
        lowest_risk_scenario = min(scenario_results.keys(),
                                 key=lambda s: scenario_results[s].risk_analysis['overall_risk_score'])
        
        comparison['key_insights'] = [
            f"Highest market value scenario: {best_value_scenario}",
            f"Lowest risk scenario: {lowest_risk_scenario}",
            f"Market value range: ${min(r.key_metrics['final_market_value_billions'] for r in scenario_results.values()):.1f}B - ${max(r.key_metrics['final_market_value_billions'] for r in scenario_results.values()):.1f}B"
        ]
        
        comparison['detailed_results'] = scenario_results
        
        return comparison
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market dynamics summary."""
        demand_summary = self.demand_model.get_demand_summary(
            self.demand_model.generate_base_demand_forecast()
        )
        supply_summary = self.supply_model.get_supply_summary()
        pricing_summary = self.pricing_model.get_pricing_summary()
        
        return {
            'forecast_horizon_years': self.forecast_horizon,
            'demand_model_summary': demand_summary,
            'supply_model_summary': supply_summary,
            'pricing_model_summary': pricing_summary,
            'integration_parameters': {
                'price_adjustment_speed': self.price_adjustment_speed,
                'capacity_response_lag': self.capacity_response_lag,
                'demand_price_feedback': self.demand_price_feedback
            },
            'available_scenarios': list(self.scenario_configs.keys())
        } 