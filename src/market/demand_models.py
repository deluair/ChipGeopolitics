"""
Demand Forecasting Models

Implements sophisticated demand forecasting for semiconductor markets including
AI-driven demand, traditional computing, automotive, IoT, and other market segments.
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

class MarketSegment(Enum):
    """Market segment types."""
    AI_DATACENTER = "ai_datacenter"
    CLOUD_COMPUTE = "cloud_compute"
    MOBILE_DEVICES = "mobile_devices"
    AUTOMOTIVE = "automotive"
    IOT_EDGE = "iot_edge"
    ENTERPRISE_COMPUTE = "enterprise_compute"
    CONSUMER_ELECTRONICS = "consumer_electronics"
    TELECOMMUNICATIONS = "telecommunications"
    INDUSTRIAL = "industrial"
    DEFENSE_AEROSPACE = "defense_aerospace"

class ProcessNodeDemand(Enum):
    """Process node demand categories."""
    LEADING_EDGE = "leading_edge"  # 3nm, 2nm
    ADVANCED = "advanced"  # 5nm, 7nm
    MAINSTREAM = "mainstream"  # 10nm, 16nm
    MATURE = "mature"  # 28nm+

@dataclass
class DemandDrivers:
    """Demand driver parameters for market segments."""
    base_growth_rate: float  # Annual base growth rate
    ai_acceleration_factor: float  # AI adoption acceleration
    economic_sensitivity: float  # Sensitivity to economic conditions
    technology_substitution_risk: float  # Risk of technology disruption
    geographic_concentration: Dict[str, float]  # Regional demand distribution
    seasonality_factor: float  # Seasonal demand variation
    price_elasticity: float  # Price elasticity of demand

@dataclass
class MarketSegmentDemand:
    """Market segment demand forecast."""
    segment: MarketSegment
    quarterly_demand_units: List[float]  # Units in millions
    quarterly_demand_value: List[float]  # Value in billions USD
    demand_drivers: DemandDrivers
    node_preference: Dict[ProcessNodeDemand, float]  # Node preference weights
    growth_trajectory: str  # "exponential", "linear", "plateau", "decline"
    uncertainty_range: Tuple[float, float]  # Confidence interval

class DemandForecastModel:
    """
    Comprehensive demand forecasting model for semiconductor markets.
    
    Integrates multiple demand drivers including:
    - AI workload growth and datacenter expansion
    - Traditional computing demand evolution
    - Automotive electrification and autonomous driving
    - IoT edge device proliferation
    - Economic cycle impacts
    - Geopolitical supply constraints
    """
    
    def __init__(self, forecast_horizon_years: int = 10):
        self.forecast_horizon = forecast_horizon_years
        self.quarters = forecast_horizon_years * 4
        
        # Market segment configurations
        self.segment_configs = self._initialize_segment_configs()
        
        # Global demand modifiers
        self.economic_cycle_impact = 0.0  # -1 to 1
        self.geopolitical_disruption = 0.0  # 0 to 1
        self.technology_disruption_events = []
        
        # Demand elasticity parameters
        self.cross_segment_elasticity = self._initialize_cross_elasticity()
        
        # Regional demand factors
        self.regional_growth_factors = {
            'North_America': 1.0,
            'Europe': 0.8,
            'Asia_Pacific': 1.3,
            'China': 1.2,
            'Rest_of_World': 1.1
        }
    
    def _initialize_segment_configs(self) -> Dict[MarketSegment, DemandDrivers]:
        """Initialize demand driver configurations for each market segment."""
        configs = {}
        
        # AI Datacenter - Explosive growth driven by AI adoption
        configs[MarketSegment.AI_DATACENTER] = DemandDrivers(
            base_growth_rate=0.35,  # 35% annual growth
            ai_acceleration_factor=1.8,
            economic_sensitivity=0.3,  # Less sensitive due to strategic importance
            technology_substitution_risk=0.2,
            geographic_concentration={
                'North_America': 0.45, 'China': 0.25, 'Europe': 0.15, 'Asia_Pacific': 0.15
            },
            seasonality_factor=0.1,
            price_elasticity=-0.3  # Relatively inelastic
        )
        
        # Cloud Compute - Steady growth with AI boost
        configs[MarketSegment.CLOUD_COMPUTE] = DemandDrivers(
            base_growth_rate=0.15,
            ai_acceleration_factor=1.3,
            economic_sensitivity=0.4,
            technology_substitution_risk=0.1,
            geographic_concentration={
                'North_America': 0.4, 'Asia_Pacific': 0.3, 'Europe': 0.2, 'China': 0.1
            },
            seasonality_factor=0.05,
            price_elasticity=-0.5
        )
        
        # Mobile Devices - Maturing market with incremental improvements
        configs[MarketSegment.MOBILE_DEVICES] = DemandDrivers(
            base_growth_rate=0.03,
            ai_acceleration_factor=1.1,
            economic_sensitivity=0.7,
            technology_substitution_risk=0.3,
            geographic_concentration={
                'Asia_Pacific': 0.4, 'China': 0.25, 'North_America': 0.2, 'Europe': 0.15
            },
            seasonality_factor=0.2,
            price_elasticity=-0.8
        )
        
        # Automotive - Rapid electrification and autonomous driving
        configs[MarketSegment.AUTOMOTIVE] = DemandDrivers(
            base_growth_rate=0.25,
            ai_acceleration_factor=1.5,
            economic_sensitivity=0.6,
            technology_substitution_risk=0.1,
            geographic_concentration={
                'China': 0.35, 'Europe': 0.25, 'North_America': 0.25, 'Asia_Pacific': 0.15
            },
            seasonality_factor=0.1,
            price_elasticity=-0.4
        )
        
        # IoT Edge - Distributed computing growth
        configs[MarketSegment.IOT_EDGE] = DemandDrivers(
            base_growth_rate=0.20,
            ai_acceleration_factor=1.4,
            economic_sensitivity=0.5,
            technology_substitution_risk=0.2,
            geographic_concentration={
                'Asia_Pacific': 0.35, 'North_America': 0.25, 'China': 0.2, 'Europe': 0.2
            },
            seasonality_factor=0.05,
            price_elasticity=-0.6
        )
        
        # Enterprise Compute - Steady replacement cycles
        configs[MarketSegment.ENTERPRISE_COMPUTE] = DemandDrivers(
            base_growth_rate=0.08,
            ai_acceleration_factor=1.2,
            economic_sensitivity=0.8,
            technology_substitution_risk=0.2,
            geographic_concentration={
                'North_America': 0.4, 'Europe': 0.25, 'Asia_Pacific': 0.2, 'China': 0.15
            },
            seasonality_factor=0.15,
            price_elasticity=-0.7
        )
        
        # Consumer Electronics - Mature market
        configs[MarketSegment.CONSUMER_ELECTRONICS] = DemandDrivers(
            base_growth_rate=0.05,
            ai_acceleration_factor=1.1,
            economic_sensitivity=0.9,
            technology_substitution_risk=0.3,
            geographic_concentration={
                'Asia_Pacific': 0.4, 'China': 0.3, 'North_America': 0.15, 'Europe': 0.15
            },
            seasonality_factor=0.3,
            price_elasticity=-1.2
        )
        
        # Telecommunications - 5G/6G infrastructure
        configs[MarketSegment.TELECOMMUNICATIONS] = DemandDrivers(
            base_growth_rate=0.12,
            ai_acceleration_factor=1.2,
            economic_sensitivity=0.4,
            technology_substitution_risk=0.15,
            geographic_concentration={
                'Asia_Pacific': 0.35, 'China': 0.25, 'North_America': 0.2, 'Europe': 0.2
            },
            seasonality_factor=0.1,
            price_elasticity=-0.4
        )
        
        # Industrial - Industry 4.0 and automation
        configs[MarketSegment.INDUSTRIAL] = DemandDrivers(
            base_growth_rate=0.10,
            ai_acceleration_factor=1.3,
            economic_sensitivity=0.7,
            technology_substitution_risk=0.2,
            geographic_concentration={
                'China': 0.3, 'Europe': 0.25, 'North_America': 0.25, 'Asia_Pacific': 0.2
            },
            seasonality_factor=0.1,
            price_elasticity=-0.5
        )
        
        # Defense/Aerospace - Strategic and specialized
        configs[MarketSegment.DEFENSE_AEROSPACE] = DemandDrivers(
            base_growth_rate=0.06,
            ai_acceleration_factor=1.4,
            economic_sensitivity=0.2,
            technology_substitution_risk=0.1,
            geographic_concentration={
                'North_America': 0.5, 'Europe': 0.2, 'Asia_Pacific': 0.15, 'China': 0.15
            },
            seasonality_factor=0.05,
            price_elasticity=-0.2
        )
        
        return configs
    
    def _initialize_cross_elasticity(self) -> Dict[Tuple[MarketSegment, MarketSegment], float]:
        """Initialize cross-elasticity between market segments."""
        # Simplified cross-elasticity matrix
        elasticity = {}
        
        # AI Datacenter and Cloud Compute have high cross-elasticity
        elasticity[(MarketSegment.AI_DATACENTER, MarketSegment.CLOUD_COMPUTE)] = 0.3
        elasticity[(MarketSegment.CLOUD_COMPUTE, MarketSegment.AI_DATACENTER)] = 0.2
        
        # Mobile and Consumer Electronics
        elasticity[(MarketSegment.MOBILE_DEVICES, MarketSegment.CONSUMER_ELECTRONICS)] = 0.4
        elasticity[(MarketSegment.CONSUMER_ELECTRONICS, MarketSegment.MOBILE_DEVICES)] = 0.3
        
        # Enterprise and Cloud Compute
        elasticity[(MarketSegment.ENTERPRISE_COMPUTE, MarketSegment.CLOUD_COMPUTE)] = 0.2
        elasticity[(MarketSegment.CLOUD_COMPUTE, MarketSegment.ENTERPRISE_COMPUTE)] = 0.1
        
        return elasticity
    
    def generate_base_demand_forecast(self, current_year: int = 2025) -> Dict[MarketSegment, MarketSegmentDemand]:
        """Generate base demand forecast for all market segments."""
        forecasts = {}
        
        # Base market sizes (2024 data in units of millions and billions USD)
        base_sizes = {
            MarketSegment.AI_DATACENTER: (50, 25),  # 50M units, $25B
            MarketSegment.CLOUD_COMPUTE: (200, 80),
            MarketSegment.MOBILE_DEVICES: (1500, 150),
            MarketSegment.AUTOMOTIVE: (100, 20),
            MarketSegment.IOT_EDGE: (800, 40),
            MarketSegment.ENTERPRISE_COMPUTE: (300, 60),
            MarketSegment.CONSUMER_ELECTRONICS: (1000, 100),
            MarketSegment.TELECOMMUNICATIONS: (150, 30),
            MarketSegment.INDUSTRIAL: (200, 25),
            MarketSegment.DEFENSE_AEROSPACE: (20, 15)
        }
        
        for segment, config in self.segment_configs.items():
            base_units, base_value = base_sizes[segment]
            
            # Generate quarterly forecasts
            quarterly_units = []
            quarterly_values = []
            
            for quarter in range(self.quarters):
                year_fraction = quarter / 4.0
                
                # Base growth with compound effect
                growth_factor = (1 + config.base_growth_rate) ** year_fraction
                
                # AI acceleration effect (stronger in early years)
                ai_boost = config.ai_acceleration_factor * math.exp(-year_fraction * 0.2)
                
                # Economic cycle impact
                economic_modifier = 1 + (self.economic_cycle_impact * config.economic_sensitivity)
                
                # Seasonality (simplified sinusoidal)
                seasonal_modifier = 1 + config.seasonality_factor * math.sin(2 * math.pi * (quarter % 4) / 4)
                
                # Technology disruption events
                disruption_modifier = self._calculate_disruption_impact(segment, quarter)
                
                # Calculate demand
                units = base_units * growth_factor * ai_boost * economic_modifier * seasonal_modifier * disruption_modifier
                
                # Value calculation with different dynamics
                value_growth = growth_factor * ai_boost * economic_modifier * seasonal_modifier
                value = base_value * value_growth
                
                quarterly_units.append(max(0, units))
                quarterly_values.append(max(0, value))
            
            # Determine growth trajectory
            final_growth = quarterly_units[-1] / quarterly_units[0] if quarterly_units[0] > 0 else 1
            if final_growth > 4:
                trajectory = "exponential"
            elif final_growth > 1.5:
                trajectory = "linear"
            elif final_growth > 0.8:
                trajectory = "plateau"
            else:
                trajectory = "decline"
            
            # Calculate uncertainty range based on volatility
            base_uncertainty = 0.15  # ±15% base uncertainty
            segment_uncertainty = base_uncertainty * (1 + config.technology_substitution_risk)
            uncertainty_range = (1 - segment_uncertainty, 1 + segment_uncertainty)
            
            # Node preference (process node demand distribution)
            node_preference = self._calculate_node_preference(segment)
            
            # Create forecast
            forecasts[segment] = MarketSegmentDemand(
                segment=segment,
                quarterly_demand_units=quarterly_units,
                quarterly_demand_value=quarterly_values,
                demand_drivers=config,
                node_preference=node_preference,
                growth_trajectory=trajectory,
                uncertainty_range=uncertainty_range
            )
        
        return forecasts
    
    def _calculate_node_preference(self, segment: MarketSegment) -> Dict[ProcessNodeDemand, float]:
        """Calculate process node preference for market segment."""
        # Node preference by market segment
        preferences = {
            MarketSegment.AI_DATACENTER: {
                ProcessNodeDemand.LEADING_EDGE: 0.6,
                ProcessNodeDemand.ADVANCED: 0.35,
                ProcessNodeDemand.MAINSTREAM: 0.05,
                ProcessNodeDemand.MATURE: 0.0
            },
            MarketSegment.CLOUD_COMPUTE: {
                ProcessNodeDemand.LEADING_EDGE: 0.3,
                ProcessNodeDemand.ADVANCED: 0.5,
                ProcessNodeDemand.MAINSTREAM: 0.2,
                ProcessNodeDemand.MATURE: 0.0
            },
            MarketSegment.MOBILE_DEVICES: {
                ProcessNodeDemand.LEADING_EDGE: 0.4,
                ProcessNodeDemand.ADVANCED: 0.45,
                ProcessNodeDemand.MAINSTREAM: 0.15,
                ProcessNodeDemand.MATURE: 0.0
            },
            MarketSegment.AUTOMOTIVE: {
                ProcessNodeDemand.LEADING_EDGE: 0.1,
                ProcessNodeDemand.ADVANCED: 0.3,
                ProcessNodeDemand.MAINSTREAM: 0.4,
                ProcessNodeDemand.MATURE: 0.2
            },
            MarketSegment.IOT_EDGE: {
                ProcessNodeDemand.LEADING_EDGE: 0.05,
                ProcessNodeDemand.ADVANCED: 0.25,
                ProcessNodeDemand.MAINSTREAM: 0.5,
                ProcessNodeDemand.MATURE: 0.2
            },
            MarketSegment.ENTERPRISE_COMPUTE: {
                ProcessNodeDemand.LEADING_EDGE: 0.2,
                ProcessNodeDemand.ADVANCED: 0.4,
                ProcessNodeDemand.MAINSTREAM: 0.3,
                ProcessNodeDemand.MATURE: 0.1
            },
            MarketSegment.CONSUMER_ELECTRONICS: {
                ProcessNodeDemand.LEADING_EDGE: 0.1,
                ProcessNodeDemand.ADVANCED: 0.2,
                ProcessNodeDemand.MAINSTREAM: 0.5,
                ProcessNodeDemand.MATURE: 0.2
            },
            MarketSegment.TELECOMMUNICATIONS: {
                ProcessNodeDemand.LEADING_EDGE: 0.2,
                ProcessNodeDemand.ADVANCED: 0.4,
                ProcessNodeDemand.MAINSTREAM: 0.3,
                ProcessNodeDemand.MATURE: 0.1
            },
            MarketSegment.INDUSTRIAL: {
                ProcessNodeDemand.LEADING_EDGE: 0.05,
                ProcessNodeDemand.ADVANCED: 0.2,
                ProcessNodeDemand.MAINSTREAM: 0.4,
                ProcessNodeDemand.MATURE: 0.35
            },
            MarketSegment.DEFENSE_AEROSPACE: {
                ProcessNodeDemand.LEADING_EDGE: 0.3,
                ProcessNodeDemand.ADVANCED: 0.4,
                ProcessNodeDemand.MAINSTREAM: 0.25,
                ProcessNodeDemand.MATURE: 0.05
            }
        }
        
        return preferences.get(segment, {
            ProcessNodeDemand.LEADING_EDGE: 0.25,
            ProcessNodeDemand.ADVANCED: 0.25,
            ProcessNodeDemand.MAINSTREAM: 0.25,
            ProcessNodeDemand.MATURE: 0.25
        })
    
    def _calculate_disruption_impact(self, segment: MarketSegment, quarter: int) -> float:
        """Calculate impact of technology disruption events."""
        impact = 1.0
        
        for event in self.technology_disruption_events:
            if event['start_quarter'] <= quarter <= event['end_quarter']:
                if segment in event['affected_segments']:
                    impact *= event['impact_factor']
        
        return impact
    
    def apply_scenario_modifiers(self, scenario: str, forecasts: Dict[MarketSegment, MarketSegmentDemand]) -> Dict[MarketSegment, MarketSegmentDemand]:
        """Apply scenario-specific modifiers to demand forecasts."""
        modifiers = self._get_scenario_modifiers(scenario)
        
        modified_forecasts = {}
        for segment, forecast in forecasts.items():
            modified_forecast = self._apply_segment_modifiers(forecast, modifiers)
            modified_forecasts[segment] = modified_forecast
        
        return modified_forecasts
    
    def _get_scenario_modifiers(self, scenario: str) -> Dict[str, float]:
        """Get demand modifiers for different scenarios."""
        scenario_configs = {
            'baseline': {
                'economic_growth': 1.0,
                'geopolitical_stability': 1.0,
                'technology_adoption': 1.0,
                'supply_chain_efficiency': 1.0
            },
            'trade_war_escalation': {
                'economic_growth': 0.8,
                'geopolitical_stability': 0.6,
                'technology_adoption': 0.9,
                'supply_chain_efficiency': 0.7
            },
            'ai_breakthrough': {
                'economic_growth': 1.2,
                'geopolitical_stability': 1.0,
                'technology_adoption': 2.0,
                'supply_chain_efficiency': 0.8
            },
            'supply_chain_crisis': {
                'economic_growth': 0.9,
                'geopolitical_stability': 0.8,
                'technology_adoption': 0.8,
                'supply_chain_efficiency': 0.5
            },
            'quantum_disruption': {
                'economic_growth': 1.1,
                'geopolitical_stability': 0.9,
                'technology_adoption': 1.5,
                'supply_chain_efficiency': 1.0
            }
        }
        
        return scenario_configs.get(scenario, scenario_configs['baseline'])
    
    def _apply_segment_modifiers(self, forecast: MarketSegmentDemand, modifiers: Dict[str, float]) -> MarketSegmentDemand:
        """Apply modifiers to individual segment forecast."""
        # Calculate combined modifier effect
        economic_impact = modifiers['economic_growth'] ** forecast.demand_drivers.economic_sensitivity
        tech_impact = modifiers['technology_adoption'] ** (1 - forecast.demand_drivers.technology_substitution_risk)
        supply_impact = modifiers['supply_chain_efficiency'] ** 0.5
        
        combined_modifier = economic_impact * tech_impact * supply_impact
        
        # Apply to quarterly forecasts
        modified_units = [units * combined_modifier for units in forecast.quarterly_demand_units]
        modified_values = [value * combined_modifier for value in forecast.quarterly_demand_value]
        
        # Adjust uncertainty range
        uncertainty_adjustment = 1 + (1 - modifiers['geopolitical_stability']) * 0.5
        old_lower, old_upper = forecast.uncertainty_range
        new_uncertainty_range = (
            old_lower * uncertainty_adjustment,
            old_upper * uncertainty_adjustment
        )
        
        # Create modified forecast
        return MarketSegmentDemand(
            segment=forecast.segment,
            quarterly_demand_units=modified_units,
            quarterly_demand_value=modified_values,
            demand_drivers=forecast.demand_drivers,
            node_preference=forecast.node_preference,
            growth_trajectory=forecast.growth_trajectory,
            uncertainty_range=new_uncertainty_range
        )
    
    def calculate_total_market_demand(self, forecasts: Dict[MarketSegment, MarketSegmentDemand]) -> Tuple[List[float], List[float]]:
        """Calculate total market demand across all segments."""
        total_units = [0] * self.quarters
        total_value = [0] * self.quarters
        
        for forecast in forecasts.values():
            for i in range(self.quarters):
                if i < len(forecast.quarterly_demand_units):
                    total_units[i] += forecast.quarterly_demand_units[i]
                if i < len(forecast.quarterly_demand_value):
                    total_value[i] += forecast.quarterly_demand_value[i]
        
        return total_units, total_value
    
    def calculate_node_demand_distribution(self, forecasts: Dict[MarketSegment, MarketSegmentDemand]) -> Dict[ProcessNodeDemand, List[float]]:
        """Calculate demand distribution by process node category."""
        node_demand = {node: [0] * self.quarters for node in ProcessNodeDemand}
        
        for forecast in forecasts.values():
            for node, preference in forecast.node_preference.items():
                for i in range(self.quarters):
                    if i < len(forecast.quarterly_demand_units):
                        node_demand[node][i] += forecast.quarterly_demand_units[i] * preference
        
        return node_demand
    
    def update_economic_conditions(self, economic_indicator: float):
        """Update economic cycle impact (-1 to 1)."""
        self.economic_cycle_impact = np.clip(economic_indicator, -1.0, 1.0)
    
    def update_geopolitical_conditions(self, disruption_level: float):
        """Update geopolitical disruption level (0 to 1)."""
        self.geopolitical_disruption = np.clip(disruption_level, 0.0, 1.0)
    
    def add_disruption_event(self, event_name: str, start_quarter: int, duration_quarters: int, 
                           affected_segments: List[MarketSegment], impact_factor: float):
        """Add a technology disruption event."""
        self.technology_disruption_events.append({
            'name': event_name,
            'start_quarter': start_quarter,
            'end_quarter': start_quarter + duration_quarters,
            'affected_segments': affected_segments,
            'impact_factor': impact_factor
        })
    
    def get_demand_summary(self, forecasts: Dict[MarketSegment, MarketSegmentDemand]) -> Dict[str, Any]:
        """Get comprehensive demand forecast summary."""
        total_units, total_value = self.calculate_total_market_demand(forecasts)
        node_distribution = self.calculate_node_demand_distribution(forecasts)
        
        # Calculate growth rates
        initial_total = sum(f.quarterly_demand_units[0] for f in forecasts.values())
        final_total = sum(f.quarterly_demand_units[-1] for f in forecasts.values())
        cagr = (final_total / initial_total) ** (1 / self.forecast_horizon) - 1 if initial_total > 0 else 0
        
        # Segment analysis
        segment_analysis = {}
        for segment, forecast in forecasts.items():
            initial_demand = forecast.quarterly_demand_units[0]
            final_demand = forecast.quarterly_demand_units[-1]
            segment_cagr = (final_demand / initial_demand) ** (1 / self.forecast_horizon) - 1 if initial_demand > 0 else 0
            
            segment_analysis[segment.value] = {
                'initial_units_millions': initial_demand,
                'final_units_millions': final_demand,
                'cagr': segment_cagr,
                'growth_trajectory': forecast.growth_trajectory,
                'market_share_initial': initial_demand / initial_total if initial_total > 0 else 0,
                'market_share_final': final_demand / final_total if final_total > 0 else 0
            }
        
        return {
            'forecast_horizon_years': self.forecast_horizon,
            'total_market_cagr': cagr,
            'initial_market_size_units': initial_total,
            'final_market_size_units': final_total,
            'initial_market_value': sum(f.quarterly_demand_value[0] for f in forecasts.values()),
            'final_market_value': sum(f.quarterly_demand_value[-1] for f in forecasts.values()),
            'segment_analysis': segment_analysis,
            'node_demand_final': {node.value: demand[-1] for node, demand in node_distribution.items()},
            'economic_conditions': self.economic_cycle_impact,
            'geopolitical_disruption': self.geopolitical_disruption,
            'active_disruption_events': len(self.technology_disruption_events)
        } 