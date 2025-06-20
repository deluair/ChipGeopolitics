"""
Energy Consumption Model for Semiconductor Manufacturing

Models energy consumption patterns, efficiency trends, and sustainability constraints
in semiconductor manufacturing including fabs, equipment, and infrastructure.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class ProcessNode(Enum):
    """Semiconductor process nodes."""
    NODE_28NM = "28nm"
    NODE_14NM = "14nm"
    NODE_10NM = "10nm"
    NODE_7NM = "7nm"
    NODE_5NM = "5nm"
    NODE_3NM = "3nm"
    NODE_2NM = "2nm"
    NODE_MATURE = "mature"  # >28nm

class EnergyIntensityCategory(Enum):
    """Energy intensity categories for different processes."""
    LOW = "low"           # <100 MWh/wafer
    MODERATE = "moderate" # 100-300 MWh/wafer
    HIGH = "high"         # 300-600 MWh/wafer
    EXTREME = "extreme"   # >600 MWh/wafer

@dataclass
class FabEnergyProfile:
    """Energy consumption profile for a semiconductor fab."""
    fab_id: str
    company: str
    location: str
    process_nodes: List[ProcessNode]
    monthly_capacity_wafers: int
    energy_consumption_mwh_per_month: float
    energy_intensity_mwh_per_wafer: float
    peak_power_demand_mw: float
    renewable_energy_percentage: float
    energy_efficiency_score: float  # 0-1 efficiency rating
    cooling_energy_percentage: float  # % of total energy for cooling
    cleanroom_energy_percentage: float  # % for cleanroom systems
    equipment_energy_breakdown: Dict[str, float]  # Equipment type -> % energy

@dataclass
class EnergyEfficiencyTrend:
    """Energy efficiency improvement trends."""
    process_node: ProcessNode
    base_year: int
    base_energy_intensity: float  # MWh per wafer
    annual_efficiency_improvement: float  # % improvement per year
    theoretical_minimum: float  # Physical limit MWh per wafer
    current_efficiency_gap: float  # Gap from theoretical minimum

@dataclass
class GridConstraint:
    """Power grid constraints and reliability."""
    region: str
    total_capacity_mw: float
    current_utilization: float  # 0-1 utilization level
    renewable_percentage: float
    reliability_score: float  # 0-1 grid reliability
    peak_demand_seasons: List[str]
    outage_frequency_per_year: float
    voltage_stability_score: float
    transmission_constraints: List[str]

class EnergyConsumptionModel:
    """
    Comprehensive energy consumption modeling for semiconductor industry.
    
    Models:
    - Fab-level energy consumption and efficiency
    - Process node energy intensity trends
    - Regional grid constraints and reliability
    - Renewable energy integration potential
    - Energy cost optimization scenarios
    - Sustainability constraint impacts
    """
    
    def __init__(self):
        # Energy consumption data
        self.fab_profiles: Dict[str, FabEnergyProfile] = {}
        self.efficiency_trends: Dict[ProcessNode, EnergyEfficiencyTrend] = {}
        self.grid_constraints: Dict[str, GridConstraint] = {}
        
        # Analysis results
        self.consumption_forecasts: Dict[str, Any] = {}
        self.efficiency_projections: Dict[str, Any] = {}
        self.grid_stress_analysis: Dict[str, Any] = {}
        
        # Initialize with realistic 2024 industry data
        self._initialize_fab_profiles()
        self._initialize_efficiency_trends()
        self._initialize_grid_constraints()
    
    def _initialize_fab_profiles(self):
        """Initialize realistic fab energy profiles."""
        
        # TSMC fabs (Taiwan)
        self.fab_profiles["tsmc_fab18"] = FabEnergyProfile(
            fab_id="tsmc_fab18",
            company="TSMC",
            location="Taiwan",
            process_nodes=[ProcessNode.NODE_3NM, ProcessNode.NODE_5NM],
            monthly_capacity_wafers=100000,
            energy_consumption_mwh_per_month=45000,  # Extremely high for advanced nodes
            energy_intensity_mwh_per_wafer=0.45,
            peak_power_demand_mw=180,
            renewable_energy_percentage=0.4,
            energy_efficiency_score=0.85,
            cooling_energy_percentage=0.35,
            cleanroom_energy_percentage=0.25,
            equipment_energy_breakdown={
                "lithography": 0.35,
                "etching": 0.20,
                "deposition": 0.15,
                "ion_implantation": 0.10,
                "metrology": 0.08,
                "other": 0.12
            }
        )
        
        self.fab_profiles["tsmc_fab14"] = FabEnergyProfile(
            fab_id="tsmc_fab14",
            company="TSMC",
            location="Taiwan",
            process_nodes=[ProcessNode.NODE_7NM, ProcessNode.NODE_10NM],
            monthly_capacity_wafers=150000,
            energy_consumption_mwh_per_month=40000,
            energy_intensity_mwh_per_wafer=0.27,
            peak_power_demand_mw=150,
            renewable_energy_percentage=0.3,
            energy_efficiency_score=0.75,
            cooling_energy_percentage=0.30,
            cleanroom_energy_percentage=0.22,
            equipment_energy_breakdown={
                "lithography": 0.30,
                "etching": 0.22,
                "deposition": 0.18,
                "ion_implantation": 0.12,
                "metrology": 0.08,
                "other": 0.10
            }
        )
        
        # Samsung fabs (South Korea)
        self.fab_profiles["samsung_s3"] = FabEnergyProfile(
            fab_id="samsung_s3",
            company="Samsung",
            location="South Korea",
            process_nodes=[ProcessNode.NODE_3NM, ProcessNode.NODE_5NM],
            monthly_capacity_wafers=80000,
            energy_consumption_mwh_per_month=38000,
            energy_intensity_mwh_per_wafer=0.48,
            peak_power_demand_mw=160,
            renewable_energy_percentage=0.25,
            energy_efficiency_score=0.80,
            cooling_energy_percentage=0.38,
            cleanroom_energy_percentage=0.24,
            equipment_energy_breakdown={
                "lithography": 0.38,
                "etching": 0.18,
                "deposition": 0.16,
                "ion_implantation": 0.10,
                "metrology": 0.08,
                "other": 0.10
            }
        )
        
        # Intel fabs (USA)
        self.fab_profiles["intel_d1x"] = FabEnergyProfile(
            fab_id="intel_d1x",
            company="Intel",
            location="Oregon, USA",
            process_nodes=[ProcessNode.NODE_7NM, ProcessNode.NODE_10NM],
            monthly_capacity_wafers=120000,
            energy_consumption_mwh_per_month=32000,
            energy_intensity_mwh_per_wafer=0.27,
            peak_power_demand_mw=130,
            renewable_energy_percentage=0.75,  # Intel's renewable commitment
            energy_efficiency_score=0.78,
            cooling_energy_percentage=0.28,
            cleanroom_energy_percentage=0.20,
            equipment_energy_breakdown={
                "lithography": 0.28,
                "etching": 0.20,
                "deposition": 0.20,
                "ion_implantation": 0.12,
                "metrology": 0.10,
                "other": 0.10
            }
        )
        
        # GlobalFoundries (USA)
        self.fab_profiles["gf_fab8"] = FabEnergyProfile(
            fab_id="gf_fab8",
            company="GlobalFoundries",
            location="New York, USA",
            process_nodes=[ProcessNode.NODE_14NM, ProcessNode.NODE_28NM],
            monthly_capacity_wafers=200000,
            energy_consumption_mwh_per_month=25000,
            energy_intensity_mwh_per_wafer=0.125,
            peak_power_demand_mw=100,
            renewable_energy_percentage=0.35,
            energy_efficiency_score=0.70,
            cooling_energy_percentage=0.25,
            cleanroom_energy_percentage=0.18,
            equipment_energy_breakdown={
                "lithography": 0.25,
                "etching": 0.22,
                "deposition": 0.20,
                "ion_implantation": 0.13,
                "metrology": 0.08,
                "other": 0.12
            }
        )
        
        # SMIC (China)
        self.fab_profiles["smic_fab1"] = FabEnergyProfile(
            fab_id="smic_fab1",
            company="SMIC",
            location="Shanghai, China",
            process_nodes=[ProcessNode.NODE_14NM, ProcessNode.NODE_28NM],
            monthly_capacity_wafers=180000,
            energy_consumption_mwh_per_month=28000,
            energy_intensity_mwh_per_wafer=0.156,
            peak_power_demand_mw=115,
            renewable_energy_percentage=0.15,
            energy_efficiency_score=0.65,
            cooling_energy_percentage=0.32,
            cleanroom_energy_percentage=0.20,
            equipment_energy_breakdown={
                "lithography": 0.22,
                "etching": 0.20,
                "deposition": 0.22,
                "ion_implantation": 0.14,
                "metrology": 0.10,
                "other": 0.12
            }
        )
    
    def _initialize_efficiency_trends(self):
        """Initialize energy efficiency improvement trends by process node."""
        
        self.efficiency_trends = {
            ProcessNode.NODE_3NM: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_3NM,
                base_year=2024,
                base_energy_intensity=0.45,  # Very high for cutting-edge
                annual_efficiency_improvement=0.08,  # 8% per year
                theoretical_minimum=0.25,  # Physical limits
                current_efficiency_gap=0.8  # Far from theoretical minimum
            ),
            
            ProcessNode.NODE_5NM: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_5NM,
                base_year=2022,
                base_energy_intensity=0.35,
                annual_efficiency_improvement=0.06,
                theoretical_minimum=0.22,
                current_efficiency_gap=0.6
            ),
            
            ProcessNode.NODE_7NM: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_7NM,
                base_year=2020,
                base_energy_intensity=0.27,
                annual_efficiency_improvement=0.05,
                theoretical_minimum=0.18,
                current_efficiency_gap=0.5
            ),
            
            ProcessNode.NODE_10NM: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_10NM,
                base_year=2018,
                base_energy_intensity=0.22,
                annual_efficiency_improvement=0.04,
                theoretical_minimum=0.15,
                current_efficiency_gap=0.4
            ),
            
            ProcessNode.NODE_14NM: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_14NM,
                base_year=2016,
                base_energy_intensity=0.15,
                annual_efficiency_improvement=0.03,
                theoretical_minimum=0.12,
                current_efficiency_gap=0.25
            ),
            
            ProcessNode.NODE_28NM: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_28NM,
                base_year=2014,
                base_energy_intensity=0.12,
                annual_efficiency_improvement=0.02,
                theoretical_minimum=0.10,
                current_efficiency_gap=0.2
            ),
            
            ProcessNode.NODE_MATURE: EnergyEfficiencyTrend(
                process_node=ProcessNode.NODE_MATURE,
                base_year=2010,
                base_energy_intensity=0.08,
                annual_efficiency_improvement=0.015,
                theoretical_minimum=0.06,
                current_efficiency_gap=0.3
            )
        }
    
    def _initialize_grid_constraints(self):
        """Initialize regional power grid constraints."""
        
        self.grid_constraints = {
            "Taiwan": GridConstraint(
                region="Taiwan",
                total_capacity_mw=55000,
                current_utilization=0.85,  # High utilization
                renewable_percentage=0.25,
                reliability_score=0.85,
                peak_demand_seasons=["summer", "winter"],
                outage_frequency_per_year=2.5,
                voltage_stability_score=0.80,
                transmission_constraints=["north_south_corridor", "island_isolation"]
            ),
            
            "South Korea": GridConstraint(
                region="South Korea",
                total_capacity_mw=130000,
                current_utilization=0.78,
                renewable_percentage=0.18,
                reliability_score=0.90,
                peak_demand_seasons=["summer"],
                outage_frequency_per_year=1.2,
                voltage_stability_score=0.88,
                transmission_constraints=["seoul_metro_congestion"]
            ),
            
            "Oregon_USA": GridConstraint(
                region="Oregon_USA",
                total_capacity_mw=25000,
                current_utilization=0.65,
                renewable_percentage=0.70,  # High hydro
                reliability_score=0.92,
                peak_demand_seasons=["winter"],
                outage_frequency_per_year=0.8,
                voltage_stability_score=0.90,
                transmission_constraints=["seasonal_hydro_variation"]
            ),
            
            "New_York_USA": GridConstraint(
                region="New_York_USA",
                total_capacity_mw=40000,
                current_utilization=0.72,
                renewable_percentage=0.35,
                reliability_score=0.88,
                peak_demand_seasons=["summer", "winter"],
                outage_frequency_per_year=1.5,
                voltage_stability_score=0.85,
                transmission_constraints=["transmission_congestion", "aging_infrastructure"]
            ),
            
            "Shanghai_China": GridConstraint(
                region="Shanghai_China",
                total_capacity_mw=35000,
                current_utilization=0.82,
                renewable_percentage=0.12,
                reliability_score=0.83,
                peak_demand_seasons=["summer"],
                outage_frequency_per_year=3.0,
                voltage_stability_score=0.78,
                transmission_constraints=["coal_dependency", "air_quality_restrictions"]
            )
        }
    
    def calculate_fab_energy_consumption(self, fab_id: str, production_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed energy consumption for a fab under production scenario."""
        
        if fab_id not in self.fab_profiles:
            return {"error": "Fab not found"}
        
        fab = self.fab_profiles[fab_id]
        
        # Base consumption calculation
        base_monthly_consumption = fab.energy_consumption_mwh_per_month
        base_capacity = fab.monthly_capacity_wafers
        
        # Scenario adjustments
        capacity_utilization = production_scenario.get("capacity_utilization", 1.0)
        process_mix = production_scenario.get("process_mix", {})  # Node -> percentage
        efficiency_improvements = production_scenario.get("efficiency_improvements", 0.0)
        
        # Calculate adjusted consumption
        utilization_factor = self._calculate_utilization_energy_factor(capacity_utilization)
        adjusted_consumption = base_monthly_consumption * utilization_factor
        
        # Process mix impact
        if process_mix:
            process_energy_factor = self._calculate_process_mix_energy_factor(fab, process_mix)
            adjusted_consumption *= process_energy_factor
        
        # Efficiency improvements
        efficiency_factor = 1.0 - efficiency_improvements
        adjusted_consumption *= efficiency_factor
        
        # Detailed breakdown
        breakdown = {
            "total_consumption_mwh": adjusted_consumption,
            "energy_intensity_mwh_per_wafer": adjusted_consumption / (base_capacity * capacity_utilization),
            "peak_demand_mw": fab.peak_power_demand_mw * utilization_factor,
            "equipment_breakdown": {
                equipment: adjusted_consumption * percentage
                for equipment, percentage in fab.equipment_energy_breakdown.items()
            },
            "infrastructure_breakdown": {
                "cooling": adjusted_consumption * fab.cooling_energy_percentage,
                "cleanroom": adjusted_consumption * fab.cleanroom_energy_percentage,
                "utilities": adjusted_consumption * 0.15,  # Estimate
                "other": adjusted_consumption * 0.10
            },
            "renewable_energy_mwh": adjusted_consumption * fab.renewable_energy_percentage,
            "grid_energy_mwh": adjusted_consumption * (1 - fab.renewable_energy_percentage),
            "carbon_intensity": self._calculate_carbon_intensity(fab, adjusted_consumption)
        }
        
        return breakdown
    
    def _calculate_utilization_energy_factor(self, utilization: float) -> float:
        """Calculate energy consumption factor based on capacity utilization."""
        # Energy doesn't scale linearly with utilization due to base loads
        base_load_factor = 0.4  # 40% energy for base systems
        variable_load_factor = 0.6  # 60% scales with production
        
        return base_load_factor + (variable_load_factor * utilization)
    
    def _calculate_process_mix_energy_factor(self, fab: FabEnergyProfile, process_mix: Dict[str, float]) -> float:
        """Calculate energy factor based on process node mix."""
        
        # Energy intensity by node (relative to baseline)
        node_energy_factors = {
            ProcessNode.NODE_3NM.value: 1.8,
            ProcessNode.NODE_5NM.value: 1.4,
            ProcessNode.NODE_7NM.value: 1.1,
            ProcessNode.NODE_10NM.value: 1.0,  # Baseline
            ProcessNode.NODE_14NM.value: 0.8,
            ProcessNode.NODE_28NM.value: 0.6,
            ProcessNode.NODE_MATURE.value: 0.4
        }
        
        weighted_factor = 0.0
        for node_str, percentage in process_mix.items():
            factor = node_energy_factors.get(node_str, 1.0)
            weighted_factor += factor * percentage
        
        return weighted_factor
    
    def _calculate_carbon_intensity(self, fab: FabEnergyProfile, consumption_mwh: float) -> Dict[str, float]:
        """Calculate carbon intensity of energy consumption."""
        
        # Grid carbon intensity by region (kg CO2/MWh)
        grid_carbon_intensity = {
            "Taiwan": 550,     # Coal-heavy grid
            "South Korea": 520,
            "Oregon, USA": 150, # Hydro-heavy
            "New York, USA": 320,
            "Shanghai, China": 650  # Coal-heavy
        }
        
        grid_intensity = grid_carbon_intensity.get(fab.location, 400)
        renewable_intensity = 50  # kg CO2/MWh for renewables (lifecycle)
        
        grid_consumption = consumption_mwh * (1 - fab.renewable_energy_percentage)
        renewable_consumption = consumption_mwh * fab.renewable_energy_percentage
        
        total_emissions = (grid_consumption * grid_intensity + 
                          renewable_consumption * renewable_intensity)
        
        return {
            "total_emissions_kg_co2": total_emissions,
            "emissions_intensity_kg_co2_per_mwh": total_emissions / consumption_mwh,
            "grid_emissions_kg_co2": grid_consumption * grid_intensity,
            "renewable_emissions_kg_co2": renewable_consumption * renewable_intensity
        }
    
    def project_efficiency_improvements(self, years: int = 10) -> Dict[str, Any]:
        """Project energy efficiency improvements over time."""
        
        projections = {}
        
        for node, trend in self.efficiency_trends.items():
            current_year = 2024
            years_since_base = current_year - trend.base_year
            
            # Current efficiency (with improvements since base year)
            current_efficiency = trend.base_energy_intensity * (
                (1 - trend.annual_efficiency_improvement) ** years_since_base
            )
            
            # Project future efficiency
            future_efficiency = []
            for year in range(years + 1):
                # Asymptotic approach to theoretical minimum
                improvement_factor = (1 - trend.annual_efficiency_improvement) ** year
                theoretical_gap_reduction = 1 - ((1 - trend.current_efficiency_gap) ** (year / 10))
                
                # Efficiency cannot go below theoretical minimum
                projected_efficiency = max(
                    trend.theoretical_minimum,
                    current_efficiency * improvement_factor * (1 - theoretical_gap_reduction * 0.1)
                )
                future_efficiency.append(projected_efficiency)
            
            projections[node.value] = {
                "current_efficiency_mwh_per_wafer": current_efficiency,
                "projected_efficiency": future_efficiency,
                "efficiency_improvement_potential": {
                    "5_year": (current_efficiency - future_efficiency[5]) / current_efficiency,
                    "10_year": (current_efficiency - future_efficiency[10]) / current_efficiency
                },
                "theoretical_minimum": trend.theoretical_minimum,
                "gap_to_minimum": current_efficiency - trend.theoretical_minimum
            }
        
        self.efficiency_projections = projections
        return projections
    
    def analyze_grid_stress(self, demand_growth_scenario: Dict[str, float]) -> Dict[str, Any]:
        """Analyze power grid stress under semiconductor demand growth."""
        
        stress_analysis = {}
        
        for region, constraint in self.grid_constraints.items():
            # Current semiconductor demand (estimate from fabs)
            current_semiconductor_demand = sum(
                fab.peak_power_demand_mw 
                for fab in self.fab_profiles.values() 
                if region.replace("_", " ") in fab.location
            )
            
            # Projected demand growth
            growth_rate = demand_growth_scenario.get(region, 0.1)  # Default 10% annual
            projected_additional_demand = current_semiconductor_demand * growth_rate
            
            # Grid stress metrics
            new_utilization = (constraint.current_utilization + 
                             (projected_additional_demand / constraint.total_capacity_mw))
            
            stress_level = self._calculate_stress_level(new_utilization)
            
            stress_analysis[region] = {
                "current_utilization": constraint.current_utilization,
                "projected_utilization": new_utilization,
                "stress_level": stress_level,
                "capacity_headroom_mw": constraint.total_capacity_mw * (1 - new_utilization),
                "reliability_risk": self._assess_reliability_risk(constraint, new_utilization),
                "infrastructure_needs": self._identify_infrastructure_needs(constraint, new_utilization),
                "renewable_integration_potential": self._assess_renewable_potential(constraint)
            }
        
        self.grid_stress_analysis = stress_analysis
        return stress_analysis
    
    def _calculate_stress_level(self, utilization: float) -> str:
        """Calculate grid stress level based on utilization."""
        if utilization < 0.7:
            return "low"
        elif utilization < 0.85:
            return "moderate"
        elif utilization < 0.95:
            return "high"
        else:
            return "critical"
    
    def _assess_reliability_risk(self, constraint: GridConstraint, new_utilization: float) -> Dict[str, Any]:
        """Assess grid reliability risks."""
        
        base_risk = 1 - constraint.reliability_score
        utilization_risk = max(0, new_utilization - 0.85) * 2  # Risk increases sharply above 85%
        
        total_risk = min(1.0, base_risk + utilization_risk)
        
        return {
            "reliability_risk_score": total_risk,
            "projected_outage_frequency": constraint.outage_frequency_per_year * (1 + utilization_risk),
            "voltage_stability_risk": max(0, new_utilization - 0.9) * 0.5,
            "peak_demand_risk": len(constraint.peak_demand_seasons) * 0.1
        }
    
    def _identify_infrastructure_needs(self, constraint: GridConstraint, new_utilization: float) -> List[str]:
        """Identify infrastructure investment needs."""
        
        needs = []
        
        if new_utilization > 0.85:
            needs.append("Generation capacity expansion")
        
        if new_utilization > 0.90:
            needs.append("Transmission system upgrades")
        
        if constraint.renewable_percentage < 0.3:
            needs.append("Renewable energy integration")
        
        if constraint.voltage_stability_score < 0.8:
            needs.append("Grid stabilization systems")
        
        if constraint.outage_frequency_per_year > 2.0:
            needs.append("Reliability improvements")
        
        return needs
    
    def _assess_renewable_potential(self, constraint: GridConstraint) -> Dict[str, float]:
        """Assess renewable energy integration potential."""
        
        return {
            "current_renewable_percentage": constraint.renewable_percentage,
            "technical_potential": min(0.8, constraint.renewable_percentage + 0.3),  # Estimate
            "economic_potential": min(0.6, constraint.renewable_percentage + 0.2),
            "policy_support_factor": 0.7,  # Generic policy support score
            "grid_flexibility_requirement": max(0.2, 1 - constraint.renewable_percentage)
        }
    
    def get_energy_summary(self) -> Dict[str, Any]:
        """Get comprehensive energy consumption summary."""
        
        # Total industry consumption
        total_monthly_consumption = sum(fab.energy_consumption_mwh_per_month 
                                      for fab in self.fab_profiles.values())
        total_annual_consumption = total_monthly_consumption * 12
        
        # Average efficiency by process node
        node_efficiencies = {}
        for fab in self.fab_profiles.values():
            for node in fab.process_nodes:
                if node.value not in node_efficiencies:
                    node_efficiencies[node.value] = []
                node_efficiencies[node.value].append(fab.energy_intensity_mwh_per_wafer)
        
        avg_node_efficiencies = {
            node: np.mean(efficiencies) 
            for node, efficiencies in node_efficiencies.items()
        }
        
        # Renewable energy adoption
        total_renewable = sum(fab.energy_consumption_mwh_per_month * fab.renewable_energy_percentage 
                            for fab in self.fab_profiles.values())
        renewable_percentage = total_renewable / total_monthly_consumption if total_monthly_consumption > 0 else 0
        
        # Grid stress levels
        if not self.grid_stress_analysis:
            self.analyze_grid_stress({})
        
        stressed_regions = [region for region, analysis in self.grid_stress_analysis.items() 
                          if analysis["stress_level"] in ["high", "critical"]]
        
        return {
            "industry_consumption": {
                "total_annual_twh": total_annual_consumption / 1000,  # Convert to TWh
                "total_fabs_analyzed": len(self.fab_profiles),
                "average_fab_consumption_mwh_per_month": total_monthly_consumption / len(self.fab_profiles),
                "renewable_energy_percentage": renewable_percentage
            },
            "efficiency_metrics": {
                "by_process_node": avg_node_efficiencies,
                "most_efficient_fab": min(self.fab_profiles.values(), 
                                        key=lambda x: x.energy_intensity_mwh_per_wafer).fab_id,
                "least_efficient_fab": max(self.fab_profiles.values(), 
                                         key=lambda x: x.energy_intensity_mwh_per_wafer).fab_id,
                "efficiency_improvement_potential": {
                    node: trend.current_efficiency_gap 
                    for node, trend in self.efficiency_trends.items()
                }
            },
            "grid_constraints": {
                "regions_analyzed": len(self.grid_constraints),
                "stressed_regions": stressed_regions,
                "average_grid_utilization": np.mean([c.current_utilization for c in self.grid_constraints.values()]),
                "renewable_integration_potential": np.mean([
                    self._assess_renewable_potential(c)["technical_potential"] 
                    for c in self.grid_constraints.values()
                ])
            },
            "sustainability_indicators": {
                "fabs_above_50_renewable": len([f for f in self.fab_profiles.values() 
                                              if f.renewable_energy_percentage > 0.5]),
                "average_cooling_energy_percentage": np.mean([f.cooling_energy_percentage 
                                                            for f in self.fab_profiles.values()]),
                "energy_efficiency_leaders": [f.fab_id for f in self.fab_profiles.values() 
                                            if f.energy_efficiency_score > 0.8]
            }
        } 