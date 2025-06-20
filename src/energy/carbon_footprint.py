"""
Carbon Footprint Analyzer for Semiconductor Industry

Comprehensive environmental impact assessment including carbon emissions,
water usage, waste generation, and circular economy integration for
semiconductor manufacturing and supply chains.
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

class EmissionScope(Enum):
    """Carbon emission scopes for reporting."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect energy emissions
    SCOPE_3 = "scope_3"  # Value chain emissions

class EmissionCategory(Enum):
    """Categories of carbon emissions."""
    ENERGY_CONSUMPTION = "energy_consumption"
    MANUFACTURING_PROCESS = "manufacturing_process"
    TRANSPORTATION = "transportation"
    RAW_MATERIALS = "raw_materials"
    WASTE_TREATMENT = "waste_treatment"
    FACILITY_OPERATIONS = "facility_operations"
    EMPLOYEE_TRAVEL = "employee_travel"

class WasteType(Enum):
    """Types of semiconductor waste."""
    CHEMICAL_WASTE = "chemical_waste"
    ELECTRONIC_WASTE = "electronic_waste"
    WAFER_SCRAP = "wafer_scrap"
    PACKAGING_WASTE = "packaging_waste"
    WATER_WASTE = "water_waste"
    HAZARDOUS_WASTE = "hazardous_waste"

@dataclass
class CarbonEmissionProfile:
    """Carbon emission profile for a facility or process."""
    entity_id: str
    entity_type: str  # "fab", "company", "supply_chain"
    location: str
    reporting_period: str  # "monthly", "quarterly", "annual"
    scope_1_emissions_tons_co2: float
    scope_2_emissions_tons_co2: float
    scope_3_emissions_tons_co2: float
    emission_intensity_kg_co2_per_wafer: float
    water_usage_cubic_meters: float
    waste_generation_tons: Dict[WasteType, float]
    renewable_energy_percentage: float
    carbon_offset_tons_co2: float

@dataclass
class EnvironmentalMetric:
    """Environmental performance metrics."""
    metric_id: str
    metric_name: str
    current_value: float
    target_value: float
    unit: str
    baseline_year: int
    improvement_trend: float  # Annual improvement rate
    regulatory_limit: Optional[float]
    industry_benchmark: float

@dataclass
class CircularEconomyMetric:
    """Circular economy integration metrics."""
    material_recovery_rate: float  # % of materials recovered/recycled
    water_recycling_rate: float   # % of water recycled
    waste_to_landfill_percentage: float
    product_lifespan_extension: float  # Years of extended use
    remanufacturing_rate: float   # % of components remanufactured
    design_for_recycling_score: float  # 0-1 design score

@dataclass
class CarbonReductionScenario:
    """Carbon reduction scenario analysis."""
    scenario_id: str
    scenario_name: str
    target_reduction_percentage: float
    timeline_years: int
    key_initiatives: List[str]
    investment_required_millions: float
    expected_cost_savings_millions: float
    technology_readiness: Dict[str, float]  # Technology -> readiness score
    implementation_risk: float  # 0-1 risk score

class CarbonFootprintAnalyzer:
    """
    Comprehensive carbon footprint and environmental impact analyzer.
    
    Analyzes:
    - Multi-scope carbon emission tracking
    - Water usage and waste generation
    - Environmental performance benchmarking
    - Carbon reduction scenario planning
    - Circular economy integration
    - Regulatory compliance assessment
    - Sustainability reporting automation
    """
    
    def __init__(self):
        # Emission and environmental data
        self.emission_profiles: Dict[str, CarbonEmissionProfile] = {}
        self.environmental_metrics: Dict[str, EnvironmentalMetric] = {}
        self.circular_economy_metrics: Dict[str, CircularEconomyMetric] = {}
        
        # Analysis results
        self.reduction_scenarios: Dict[str, CarbonReductionScenario] = {}
        self.benchmark_analysis: Dict[str, Any] = {}
        self.compliance_assessment: Dict[str, Any] = {}
        
        # Initialize with realistic 2024 industry data
        self._initialize_emission_profiles()
        self._initialize_environmental_metrics()
        self._initialize_circular_economy_data()
    
    def _initialize_emission_profiles(self):
        """Initialize realistic carbon emission profiles for major fabs."""
        
        # TSMC advanced fab (3nm/5nm) - High emissions due to energy intensity
        self.emission_profiles["tsmc_fab18"] = CarbonEmissionProfile(
            entity_id="tsmc_fab18",
            entity_type="fab",
            location="Taiwan",
            reporting_period="monthly",
            scope_1_emissions_tons_co2=2500,    # Direct emissions (fuel, chemicals)
            scope_2_emissions_tons_co2=18000,   # Electricity (coal-heavy Taiwan grid)
            scope_3_emissions_tons_co2=8500,    # Supply chain, transport
            emission_intensity_kg_co2_per_wafer=290,  # Very high for advanced nodes
            water_usage_cubic_meters=850000,    # High water usage
            waste_generation_tons={
                WasteType.CHEMICAL_WASTE: 450,
                WasteType.WAFER_SCRAP: 120,
                WasteType.WATER_WASTE: 750,
                WasteType.HAZARDOUS_WASTE: 85,
                WasteType.PACKAGING_WASTE: 25
            },
            renewable_energy_percentage=0.4,
            carbon_offset_tons_co2=3000
        )
        
        # Samsung advanced fab
        self.emission_profiles["samsung_s3"] = CarbonEmissionProfile(
            entity_id="samsung_s3",
            entity_type="fab",
            location="South Korea",
            reporting_period="monthly",
            scope_1_emissions_tons_co2=2200,
            scope_2_emissions_tons_co2=16500,   # Coal-heavy Korean grid
            scope_3_emissions_tons_co2=7800,
            emission_intensity_kg_co2_per_wafer=315,  # Slightly higher than TSMC
            water_usage_cubic_meters=780000,
            waste_generation_tons={
                WasteType.CHEMICAL_WASTE: 420,
                WasteType.WAFER_SCRAP: 95,
                WasteType.WATER_WASTE: 680,
                WasteType.HAZARDOUS_WASTE: 75,
                WasteType.PACKAGING_WASTE: 30
            },
            renewable_energy_percentage=0.25,
            carbon_offset_tons_co2=2500
        )
        
        # Intel Oregon fab - Better renewable energy
        self.emission_profiles["intel_d1x"] = CarbonEmissionProfile(
            entity_id="intel_d1x",
            entity_type="fab",
            location="Oregon, USA",
            reporting_period="monthly",
            scope_1_emissions_tons_co2=1800,
            scope_2_emissions_tons_co2=6500,    # Much lower due to hydro power
            scope_3_emissions_tons_co2=7200,
            emission_intensity_kg_co2_per_wafer=145,  # Much lower due to clean energy
            water_usage_cubic_meters=620000,
            waste_generation_tons={
                WasteType.CHEMICAL_WASTE: 380,
                WasteType.WAFER_SCRAP: 110,
                WasteType.WATER_WASTE: 580,
                WasteType.HAZARDOUS_WASTE: 65,
                WasteType.PACKAGING_WASTE: 35
            },
            renewable_energy_percentage=0.75,  # Intel's renewable commitment
            carbon_offset_tons_co2=4000
        )
        
        # GlobalFoundries mature node fab
        self.emission_profiles["gf_fab8"] = CarbonEmissionProfile(
            entity_id="gf_fab8",
            entity_type="fab",
            location="New York, USA",
            reporting_period="monthly",
            scope_1_emissions_tons_co2=1400,
            scope_2_emissions_tons_co2=9500,
            scope_3_emissions_tons_co2=5200,
            emission_intensity_kg_co2_per_wafer=85,   # Lower for mature nodes
            water_usage_cubic_meters=480000,
            waste_generation_tons={
                WasteType.CHEMICAL_WASTE: 320,
                WasteType.WAFER_SCRAP: 180,  # Higher scrap for mature processes
                WasteType.WATER_WASTE: 450,
                WasteType.HAZARDOUS_WASTE: 55,
                WasteType.PACKAGING_WASTE: 40
            },
            renewable_energy_percentage=0.35,
            carbon_offset_tons_co2=1800
        )
        
        # SMIC China fab - Coal-heavy grid
        self.emission_profiles["smic_fab1"] = CarbonEmissionProfile(
            entity_id="smic_fab1",
            entity_type="fab",
            location="Shanghai, China",
            reporting_period="monthly",
            scope_1_emissions_tons_co2=2000,
            scope_2_emissions_tons_co2=22000,   # Very high due to coal
            scope_3_emissions_tons_co2=6800,
            emission_intensity_kg_co2_per_wafer=175,
            water_usage_cubic_meters=720000,
            waste_generation_tons={
                WasteType.CHEMICAL_WASTE: 390,
                WasteType.WAFER_SCRAP: 140,
                WasteType.WATER_WASTE: 650,
                WasteType.HAZARDOUS_WASTE: 70,
                WasteType.PACKAGING_WASTE: 28
            },
            renewable_energy_percentage=0.15,  # Low renewable adoption
            carbon_offset_tons_co2=1200
        )
    
    def _initialize_environmental_metrics(self):
        """Initialize key environmental performance metrics."""
        
        # Carbon intensity improvement
        self.environmental_metrics["carbon_intensity"] = EnvironmentalMetric(
            metric_id="carbon_intensity",
            metric_name="Carbon Intensity per Wafer",
            current_value=200,  # kg CO2/wafer average
            target_value=120,   # 40% reduction target
            unit="kg CO2/wafer",
            baseline_year=2020,
            improvement_trend=0.08,  # 8% annual improvement
            regulatory_limit=None,
            industry_benchmark=185
        )
        
        # Water usage efficiency
        self.environmental_metrics["water_intensity"] = EnvironmentalMetric(
            metric_id="water_intensity",
            metric_name="Water Usage per Wafer",
            current_value=4.2,  # cubic meters per wafer
            target_value=3.0,   # 30% reduction target
            unit="m³/wafer",
            baseline_year=2020,
            improvement_trend=0.06,
            regulatory_limit=None,
            industry_benchmark=3.8
        )
        
        # Waste generation
        self.environmental_metrics["waste_intensity"] = EnvironmentalMetric(
            metric_id="waste_intensity",
            metric_name="Waste Generation per Wafer",
            current_value=2.8,  # kg waste per wafer
            target_value=2.0,
            unit="kg/wafer",
            baseline_year=2020,
            improvement_trend=0.05,
            regulatory_limit=None,
            industry_benchmark=2.5
        )
        
        # Renewable energy adoption
        self.environmental_metrics["renewable_energy"] = EnvironmentalMetric(
            metric_id="renewable_energy",
            metric_name="Renewable Energy Percentage",
            current_value=35,   # % renewable
            target_value=75,    # Industry target
            unit="percentage",
            baseline_year=2020,
            improvement_trend=0.12,  # 12% annual increase
            regulatory_limit=None,
            industry_benchmark=45
        )
        
        # Chemical usage efficiency
        self.environmental_metrics["chemical_efficiency"] = EnvironmentalMetric(
            metric_id="chemical_efficiency",
            metric_name="Chemical Usage per Wafer",
            current_value=1.5,  # kg chemicals per wafer
            target_value=1.0,
            unit="kg/wafer",
            baseline_year=2020,
            improvement_trend=0.07,
            regulatory_limit=2.0,  # Hypothetical regulatory limit
            industry_benchmark=1.3
        )
    
    def _initialize_circular_economy_data(self):
        """Initialize circular economy metrics for major companies."""
        
        # Industry leaders in circular economy
        self.circular_economy_metrics["tsmc"] = CircularEconomyMetric(
            material_recovery_rate=0.65,      # 65% materials recovered
            water_recycling_rate=0.85,        # 85% water recycled
            waste_to_landfill_percentage=0.08, # 8% to landfill
            product_lifespan_extension=2.5,   # 2.5 years extended use
            remanufacturing_rate=0.35,        # 35% components remanufactured
            design_for_recycling_score=0.7    # 70% design score
        )
        
        self.circular_economy_metrics["samsung"] = CircularEconomyMetric(
            material_recovery_rate=0.62,
            water_recycling_rate=0.80,
            waste_to_landfill_percentage=0.12,
            product_lifespan_extension=2.0,
            remanufacturing_rate=0.30,
            design_for_recycling_score=0.65
        )
        
        self.circular_economy_metrics["intel"] = CircularEconomyMetric(
            material_recovery_rate=0.70,      # Intel leads in some areas
            water_recycling_rate=0.90,        # Very high water recycling
            waste_to_landfill_percentage=0.05, # Excellent waste management
            product_lifespan_extension=3.0,
            remanufacturing_rate=0.40,
            design_for_recycling_score=0.75
        )
        
        # Industry average
        self.circular_economy_metrics["industry_average"] = CircularEconomyMetric(
            material_recovery_rate=0.45,
            water_recycling_rate=0.65,
            waste_to_landfill_percentage=0.25,
            product_lifespan_extension=1.5,
            remanufacturing_rate=0.20,
            design_for_recycling_score=0.50
        )
    
    def calculate_total_footprint(self, entity_id: str, time_period: str = "annual") -> Dict[str, Any]:
        """Calculate comprehensive carbon footprint for an entity."""
        
        if entity_id not in self.emission_profiles:
            return {"error": "Entity not found"}
        
        profile = self.emission_profiles[entity_id]
        
        # Time period multiplier
        if time_period == "annual":
            multiplier = 12 if profile.reporting_period == "monthly" else 4 if profile.reporting_period == "quarterly" else 1
        elif time_period == "monthly":
            multiplier = 1 if profile.reporting_period == "monthly" else 0.33 if profile.reporting_period == "quarterly" else 0.083
        else:
            multiplier = 1
        
        # Calculate total emissions
        total_scope_1 = profile.scope_1_emissions_tons_co2 * multiplier
        total_scope_2 = profile.scope_2_emissions_tons_co2 * multiplier
        total_scope_3 = profile.scope_3_emissions_tons_co2 * multiplier
        total_emissions = total_scope_1 + total_scope_2 + total_scope_3
        
        # Net emissions after offsets
        net_emissions = total_emissions - (profile.carbon_offset_tons_co2 * multiplier)
        
        # Environmental impact breakdown
        footprint_analysis = {
            "entity_id": entity_id,
            "time_period": time_period,
            "carbon_emissions": {
                "scope_1_tons_co2": total_scope_1,
                "scope_2_tons_co2": total_scope_2,
                "scope_3_tons_co2": total_scope_3,
                "total_gross_emissions": total_emissions,
                "carbon_offsets": profile.carbon_offset_tons_co2 * multiplier,
                "net_emissions": net_emissions,
                "emission_intensity_kg_per_wafer": profile.emission_intensity_kg_co2_per_wafer
            },
            "resource_consumption": {
                "water_usage_cubic_meters": profile.water_usage_cubic_meters * multiplier,
                "renewable_energy_percentage": profile.renewable_energy_percentage,
                "grid_energy_emissions": total_scope_2,
                "renewable_energy_emissions": total_scope_2 * (profile.renewable_energy_percentage * 0.1)  # 10% of grid intensity
            },
            "waste_generation": {
                waste_type.value: amount * multiplier 
                for waste_type, amount in profile.waste_generation_tons.items()
            },
            "environmental_impact_scores": self._calculate_impact_scores(profile, multiplier),
            "benchmarking": self._benchmark_against_industry(profile)
        }
        
        return footprint_analysis
    
    def _calculate_impact_scores(self, profile: CarbonEmissionProfile, multiplier: float) -> Dict[str, float]:
        """Calculate normalized environmental impact scores."""
        
        # Normalize against industry benchmarks (0-1 scale, lower is better)
        total_emissions = (profile.scope_1_emissions_tons_co2 + 
                          profile.scope_2_emissions_tons_co2 + 
                          profile.scope_3_emissions_tons_co2) * multiplier
        
        # Industry benchmarks (annual basis)
        benchmark_emissions = 180000  # tons CO2 per year for typical fab
        benchmark_water = 7200000     # cubic meters per year
        benchmark_waste = 1500        # tons per year
        
        carbon_score = min(1.0, total_emissions / benchmark_emissions)
        water_score = min(1.0, (profile.water_usage_cubic_meters * multiplier) / benchmark_water)
        waste_score = min(1.0, sum(profile.waste_generation_tons.values()) * multiplier / benchmark_waste)
        
        # Renewable energy score (higher is better, so invert)
        renewable_score = 1.0 - profile.renewable_energy_percentage
        
        return {
            "carbon_impact_score": carbon_score,
            "water_impact_score": water_score,
            "waste_impact_score": waste_score,
            "renewable_energy_score": renewable_score,
            "overall_impact_score": (carbon_score + water_score + waste_score + renewable_score) / 4
        }
    
    def _benchmark_against_industry(self, profile: CarbonEmissionProfile) -> Dict[str, str]:
        """Benchmark performance against industry standards."""
        
        benchmarks = {}
        
        # Carbon intensity benchmark
        if profile.emission_intensity_kg_co2_per_wafer < 150:
            benchmarks["carbon_intensity"] = "Excellent"
        elif profile.emission_intensity_kg_co2_per_wafer < 200:
            benchmarks["carbon_intensity"] = "Good"
        elif profile.emission_intensity_kg_co2_per_wafer < 250:
            benchmarks["carbon_intensity"] = "Average"
        else:
            benchmarks["carbon_intensity"] = "Below Average"
        
        # Renewable energy benchmark
        if profile.renewable_energy_percentage > 0.6:
            benchmarks["renewable_energy"] = "Excellent"
        elif profile.renewable_energy_percentage > 0.4:
            benchmarks["renewable_energy"] = "Good"
        elif profile.renewable_energy_percentage > 0.2:
            benchmarks["renewable_energy"] = "Average"
        else:
            benchmarks["renewable_energy"] = "Below Average"
        
        # Water usage benchmark (cubic meters per month)
        if profile.water_usage_cubic_meters < 500000:
            benchmarks["water_efficiency"] = "Excellent"
        elif profile.water_usage_cubic_meters < 700000:
            benchmarks["water_efficiency"] = "Good"
        elif profile.water_usage_cubic_meters < 900000:
            benchmarks["water_efficiency"] = "Average"
        else:
            benchmarks["water_efficiency"] = "Below Average"
        
        return benchmarks
    
    def analyze_reduction_potential(self, entity_id: str, target_reduction: float) -> Dict[str, Any]:
        """Analyze carbon reduction potential and pathways."""
        
        if entity_id not in self.emission_profiles:
            return {"error": "Entity not found"}
        
        profile = self.emission_profiles[entity_id]
        current_annual_emissions = (profile.scope_1_emissions_tons_co2 + 
                                   profile.scope_2_emissions_tons_co2 + 
                                   profile.scope_3_emissions_tons_co2) * 12
        
        target_emissions = current_annual_emissions * (1 - target_reduction)
        reduction_needed = current_annual_emissions - target_emissions
        
        # Reduction pathway analysis
        pathways = {
            "renewable_energy_transition": {
                "potential_reduction_tons": profile.scope_2_emissions_tons_co2 * 12 * 0.9,  # 90% of scope 2
                "investment_required_millions": 150,
                "timeline_years": 5,
                "feasibility_score": 0.8
            },
            "energy_efficiency_improvements": {
                "potential_reduction_tons": current_annual_emissions * 0.15,  # 15% reduction
                "investment_required_millions": 80,
                "timeline_years": 3,
                "feasibility_score": 0.9
            },
            "process_optimization": {
                "potential_reduction_tons": profile.scope_1_emissions_tons_co2 * 12 * 0.3,  # 30% of scope 1
                "investment_required_millions": 120,
                "timeline_years": 4,
                "feasibility_score": 0.7
            },
            "supply_chain_optimization": {
                "potential_reduction_tons": profile.scope_3_emissions_tons_co2 * 12 * 0.25,  # 25% of scope 3
                "investment_required_millions": 60,
                "timeline_years": 6,
                "feasibility_score": 0.6
            },
            "carbon_capture_storage": {
                "potential_reduction_tons": current_annual_emissions * 0.1,  # 10% reduction
                "investment_required_millions": 300,
                "timeline_years": 8,
                "feasibility_score": 0.4
            }
        }
        
        # Calculate optimal pathway combination
        optimal_combination = self._optimize_reduction_pathways(pathways, reduction_needed, target_reduction)
        
        return {
            "entity_id": entity_id,
            "current_annual_emissions": current_annual_emissions,
            "target_reduction_percentage": target_reduction,
            "reduction_needed_tons": reduction_needed,
            "available_pathways": pathways,
            "optimal_combination": optimal_combination,
            "timeline_assessment": self._assess_reduction_timeline(pathways, optimal_combination),
            "investment_summary": self._calculate_investment_summary(pathways, optimal_combination)
        }
    
    def _optimize_reduction_pathways(self, pathways: Dict[str, Dict], reduction_needed: float, 
                                   target_reduction: float) -> Dict[str, Any]:
        """Optimize combination of reduction pathways."""
        
        # Sort pathways by cost-effectiveness (reduction per dollar)
        pathway_efficiency = {}
        for name, pathway in pathways.items():
            efficiency = pathway["potential_reduction_tons"] / pathway["investment_required_millions"]
            pathway_efficiency[name] = {
                "efficiency_tons_per_million": efficiency,
                "pathway_data": pathway
            }
        
        # Select pathways starting with most efficient
        sorted_pathways = sorted(pathway_efficiency.items(), 
                               key=lambda x: x[1]["efficiency_tons_per_million"], reverse=True)
        
        selected_pathways = {}
        total_reduction = 0
        total_investment = 0
        
        for name, data in sorted_pathways:
            pathway = data["pathway_data"]
            
            # Add pathway if we haven't reached target and it's feasible
            if total_reduction < reduction_needed and pathway["feasibility_score"] > 0.5:
                utilization_factor = min(1.0, 
                    (reduction_needed - total_reduction) / pathway["potential_reduction_tons"])
                
                selected_pathways[name] = {
                    "utilization_percentage": utilization_factor,
                    "reduction_contribution": pathway["potential_reduction_tons"] * utilization_factor,
                    "investment_required": pathway["investment_required_millions"] * utilization_factor,
                    "timeline_years": pathway["timeline_years"],
                    "feasibility_score": pathway["feasibility_score"]
                }
                
                total_reduction += pathway["potential_reduction_tons"] * utilization_factor
                total_investment += pathway["investment_required_millions"] * utilization_factor
        
        achievement_percentage = min(1.0, total_reduction / reduction_needed)
        
        return {
            "selected_pathways": selected_pathways,
            "total_reduction_tons": total_reduction,
            "total_investment_millions": total_investment,
            "target_achievement_percentage": achievement_percentage,
            "cost_per_ton_reduced": total_investment * 1000000 / total_reduction if total_reduction > 0 else 0
        }
    
    def _assess_reduction_timeline(self, pathways: Dict, optimal_combination: Dict) -> Dict[str, Any]:
        """Assess implementation timeline for reduction plan."""
        
        selected = optimal_combination["selected_pathways"]
        
        if not selected:
            return {"error": "No pathways selected"}
        
        # Timeline analysis
        shortest_timeline = min(pathways[name]["timeline_years"] for name in selected.keys())
        longest_timeline = max(pathways[name]["timeline_years"] for name in selected.keys())
        weighted_avg_timeline = sum(
            pathways[name]["timeline_years"] * data["utilization_percentage"]
            for name, data in selected.items()
        ) / len(selected)
        
        # Implementation phases
        phases = {
            "immediate": [],  # 0-2 years
            "short_term": [], # 2-4 years
            "medium_term": [], # 4-7 years
            "long_term": []   # 7+ years
        }
        
        for name, data in selected.items():
            timeline = pathways[name]["timeline_years"]
            if timeline <= 2:
                phases["immediate"].append(name)
            elif timeline <= 4:
                phases["short_term"].append(name)
            elif timeline <= 7:
                phases["medium_term"].append(name)
            else:
                phases["long_term"].append(name)
        
        return {
            "shortest_pathway_years": shortest_timeline,
            "longest_pathway_years": longest_timeline,
            "weighted_average_timeline": weighted_avg_timeline,
            "implementation_phases": phases,
            "early_wins_potential": len(phases["immediate"]) + len(phases["short_term"]),
            "complexity_score": len(selected) * 0.2 + longest_timeline * 0.1
        }
    
    def _calculate_investment_summary(self, pathways: Dict, optimal_combination: Dict) -> Dict[str, Any]:
        """Calculate investment requirements and returns."""
        
        selected = optimal_combination["selected_pathways"]
        total_investment = optimal_combination["total_investment_millions"]
        total_reduction = optimal_combination["total_reduction_tons"]
        
        # Estimate cost savings from efficiency improvements and carbon pricing
        annual_energy_savings = total_reduction * 0.3 * 50  # $50/ton CO2 equivalent energy savings
        carbon_price_savings = total_reduction * 75  # $75/ton CO2 carbon price
        total_annual_savings = annual_energy_savings + carbon_price_savings
        
        payback_period = total_investment * 1000000 / total_annual_savings if total_annual_savings > 0 else float('inf')
        
        return {
            "total_investment_millions": total_investment,
            "annual_savings_dollars": total_annual_savings,
            "payback_period_years": payback_period,
            "roi_percentage": (total_annual_savings * 10 - total_investment * 1000000) / (total_investment * 1000000) * 100,
            "carbon_price_assumption": 75,  # $/ton CO2
            "investment_breakdown": {
                name: data["investment_required"] 
                for name, data in selected.items()
            }
        }
    
    def get_carbon_summary(self) -> Dict[str, Any]:
        """Get comprehensive carbon footprint summary."""
        
        # Total industry emissions
        total_emissions = {}
        total_water = 0
        total_waste = 0
        renewable_weighted_avg = 0
        
        for entity_id, profile in self.emission_profiles.items():
            annual_emissions = (profile.scope_1_emissions_tons_co2 + 
                              profile.scope_2_emissions_tons_co2 + 
                              profile.scope_3_emissions_tons_co2) * 12
            total_emissions[entity_id] = annual_emissions
            total_water += profile.water_usage_cubic_meters * 12
            total_waste += sum(profile.waste_generation_tons.values()) * 12
            renewable_weighted_avg += profile.renewable_energy_percentage
        
        if self.emission_profiles:
            renewable_weighted_avg /= len(self.emission_profiles)
        
        # Performance leaders and laggards
        if total_emissions:
            best_performer = min(total_emissions.items(), key=lambda x: x[1])
            worst_performer = max(total_emissions.items(), key=lambda x: x[1])
        else:
            best_performer = worst_performer = (None, 0)
        
        # Reduction potential analysis
        total_industry_emissions = sum(total_emissions.values())
        theoretical_minimum = total_industry_emissions * 0.4  # 60% reduction theoretical max
        
        return {
            "industry_overview": {
                "total_annual_emissions_tons_co2": total_industry_emissions,
                "total_annual_water_usage_cubic_meters": total_water,
                "total_annual_waste_tons": total_waste,
                "average_renewable_energy_percentage": renewable_weighted_avg,
                "facilities_analyzed": len(self.emission_profiles)
            },
            "performance_benchmarking": {
                "best_performer": best_performer[0],
                "worst_performer": worst_performer[0],
                "performance_gap_ratio": worst_performer[1] / best_performer[1] if best_performer[1] > 0 else 1,
                "industry_leaders": [entity for entity, profile in self.emission_profiles.items() 
                                   if profile.renewable_energy_percentage > 0.5]
            },
            "reduction_potential": {
                "current_emissions": total_industry_emissions,
                "theoretical_minimum": theoretical_minimum,
                "maximum_reduction_potential": total_industry_emissions - theoretical_minimum,
                "reduction_percentage_potential": (total_industry_emissions - theoretical_minimum) / total_industry_emissions * 100
            },
            "circular_economy_status": {
                "companies_with_metrics": len(self.circular_economy_metrics),
                "average_material_recovery": np.mean([m.material_recovery_rate for m in self.circular_economy_metrics.values()]),
                "average_water_recycling": np.mean([m.water_recycling_rate for m in self.circular_economy_metrics.values()]),
                "leaders_in_circularity": [name for name, metrics in self.circular_economy_metrics.items() 
                                         if metrics.material_recovery_rate > 0.6 and name != "industry_average"]
            },
            "environmental_targets": {
                metric_id: {
                    "current_value": metric.current_value,
                    "target_value": metric.target_value,
                    "progress_percentage": ((metric.current_value - metric.target_value) / 
                                          (metric.current_value - metric.target_value)) * 100 
                                          if metric.current_value != metric.target_value else 100,
                    "on_track": metric.improvement_trend > 0
                }
                for metric_id, metric in self.environmental_metrics.items()
            }
        } 