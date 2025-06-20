"""
Economic Impact Model for Energy and Environmental Policies

Comprehensive economic impact assessment for energy transitions, 
environmental regulations, and sustainability initiatives in the 
semiconductor industry.
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

class PolicyType(Enum):
    """Types of energy and environmental policies."""
    CARBON_TAX = "carbon_tax"
    EMISSIONS_TRADING = "emissions_trading"
    RENEWABLE_MANDATE = "renewable_mandate"
    ENERGY_EFFICIENCY_STANDARD = "energy_efficiency_standard"
    WASTE_REDUCTION_TARGET = "waste_reduction_target"
    WATER_USAGE_LIMIT = "water_usage_limit"
    TECH_INNOVATION_INCENTIVE = "tech_innovation_incentive"
    CIRCULAR_ECONOMY_REGULATION = "circular_economy_regulation"

class ImpactCategory(Enum):
    """Categories of economic impact."""
    OPERATIONAL_COSTS = "operational_costs"
    CAPITAL_INVESTMENTS = "capital_investments"
    COMPLIANCE_COSTS = "compliance_costs"
    REVENUE_IMPACTS = "revenue_impacts"
    MARKET_COMPETITIVENESS = "market_competitiveness"
    INNOVATION_BENEFITS = "innovation_benefits"
    RISK_MITIGATION = "risk_mitigation"

class Stakeholder(Enum):
    """Stakeholder types for impact analysis."""
    SEMICONDUCTOR_MANUFACTURERS = "semiconductor_manufacturers"
    EQUIPMENT_SUPPLIERS = "equipment_suppliers"
    ENERGY_PROVIDERS = "energy_providers"
    GOVERNMENTS = "governments"
    CONSUMERS = "consumers"
    INVESTORS = "investors"
    COMMUNITIES = "communities"

@dataclass
class PolicyScenario:
    """Economic policy scenario definition."""
    scenario_id: str
    scenario_name: str
    policy_type: PolicyType
    implementation_timeline: Dict[str, float]  # Year -> intensity
    geographic_scope: List[str]  # Countries/regions
    affected_sectors: List[str]
    policy_stringency: float  # 0-1 scale
    exemptions_and_flexibility: Dict[str, float]

@dataclass
class EconomicImpact:
    """Economic impact assessment results."""
    impact_id: str
    policy_scenario: str
    stakeholder: Stakeholder
    impact_category: ImpactCategory
    annual_cost_millions: float
    annual_benefit_millions: float
    net_impact_millions: float
    implementation_cost_millions: float
    payback_period_years: float
    uncertainty_range: Tuple[float, float]  # (low, high) confidence bounds
    regional_variations: Dict[str, float]  # Region -> impact multiplier

@dataclass
class CompetitivePosition:
    """Competitive positioning analysis."""
    company_id: str
    baseline_market_share: float
    projected_market_share: float
    cost_competitiveness_score: float  # 0-1, higher is better
    innovation_readiness_score: float
    regulatory_compliance_score: float
    overall_competitiveness_change: float  # Percentage change

@dataclass
class MacroeconomicEffect:
    """Macroeconomic effects of policy implementation."""
    gdp_impact_percentage: float
    employment_impact_jobs: int
    trade_balance_impact_millions: float
    innovation_investment_change_percentage: float
    energy_security_score_change: float
    regional_development_impact: Dict[str, float]

class EconomicImpactModel:
    """
    Comprehensive economic impact model for energy and environmental policies.
    
    Analyzes:
    - Multi-stakeholder economic impact assessment
    - Competitive positioning changes
    - Cost-benefit analysis with uncertainty quantification
    - Regional and sectoral impact variations
    - Innovation and investment effects
    - Macroeconomic consequences
    - Policy design optimization
    """
    
    def __init__(self):
        # Policy and impact data
        self.policy_scenarios: Dict[str, PolicyScenario] = {}
        self.economic_impacts: Dict[str, List[EconomicImpact]] = {}
        self.competitive_positions: Dict[str, CompetitivePosition] = {}
        
        # Analysis results
        self.scenario_comparisons: Dict[str, Any] = {}
        self.optimal_policy_designs: Dict[str, Any] = {}
        self.macroeconomic_effects: Dict[str, MacroeconomicEffect] = {}
        
        # Initialize with realistic 2024 policy scenarios
        self._initialize_policy_scenarios()
        self._initialize_baseline_economics()
        self._initialize_competitive_landscape()
    
    def _initialize_policy_scenarios(self):
        """Initialize realistic energy and environmental policy scenarios."""
        
        # Aggressive carbon tax scenario
        self.policy_scenarios["carbon_tax_aggressive"] = PolicyScenario(
            scenario_id="carbon_tax_aggressive",
            scenario_name="Aggressive Carbon Tax Implementation",
            policy_type=PolicyType.CARBON_TAX,
            implementation_timeline={
                "2024": 0.2,  # $50/ton CO2
                "2025": 0.4,  # $100/ton CO2
                "2026": 0.6,  # $150/ton CO2
                "2027": 0.8,  # $200/ton CO2
                "2028": 1.0   # $250/ton CO2
            },
            geographic_scope=["USA", "EU", "Japan", "South Korea"],
            affected_sectors=["semiconductor_manufacturing", "equipment_supply", "energy"],
            policy_stringency=0.9,
            exemptions_and_flexibility={"small_companies": 0.5, "early_adopters": 0.3}
        )
        
        # Moderate renewable energy mandate
        self.policy_scenarios["renewable_mandate_moderate"] = PolicyScenario(
            scenario_id="renewable_mandate_moderate",
            scenario_name="75% Renewable Energy Mandate by 2030",
            policy_type=PolicyType.RENEWABLE_MANDATE,
            implementation_timeline={
                "2024": 0.3,  # 30% minimum renewable
                "2025": 0.45, # 45% minimum renewable
                "2026": 0.6,  # 60% minimum renewable
                "2028": 0.75  # 75% minimum renewable
            },
            geographic_scope=["EU", "California", "Taiwan", "South Korea"],
            affected_sectors=["semiconductor_manufacturing", "data_centers"],
            policy_stringency=0.7,
            exemptions_and_flexibility={"grid_constraints": 0.8, "cost_hardship": 0.6}
        )
        
        # Energy efficiency standards
        self.policy_scenarios["efficiency_standards"] = PolicyScenario(
            scenario_id="efficiency_standards",
            scenario_name="Mandatory Energy Efficiency Improvements",
            policy_type=PolicyType.ENERGY_EFFICIENCY_STANDARD,
            implementation_timeline={
                "2024": 0.05, # 5% improvement required
                "2025": 0.1,  # 10% cumulative improvement
                "2026": 0.15, # 15% cumulative improvement
                "2027": 0.2,  # 20% cumulative improvement
                "2028": 0.25  # 25% cumulative improvement
            },
            geographic_scope=["USA", "EU", "China", "Japan", "South Korea", "Taiwan"],
            affected_sectors=["semiconductor_manufacturing", "equipment_supply"],
            policy_stringency=0.6,
            exemptions_and_flexibility={"legacy_fabs": 0.7, "R&D_facilities": 0.5}
        )
        
        # Circular economy regulation
        self.policy_scenarios["circular_economy"] = PolicyScenario(
            scenario_id="circular_economy",
            scenario_name="Comprehensive Circular Economy Regulation",
            policy_type=PolicyType.CIRCULAR_ECONOMY_REGULATION,
            implementation_timeline={
                "2024": 0.25, # 25% recycling/recovery target
                "2025": 0.4,  # 40% recycling/recovery target
                "2026": 0.55, # 55% recycling/recovery target
                "2027": 0.7,  # 70% recycling/recovery target
                "2028": 0.8   # 80% recycling/recovery target
            },
            geographic_scope=["EU", "Japan", "South Korea"],
            affected_sectors=["semiconductor_manufacturing", "equipment_supply", "electronics"],
            policy_stringency=0.8,
            exemptions_and_flexibility={"technology_constraints": 0.6, "infrastructure_limits": 0.7}
        )
        
        # Innovation incentive program
        self.policy_scenarios["innovation_incentives"] = PolicyScenario(
            scenario_id="innovation_incentives",
            scenario_name="Green Technology Innovation Incentives",
            policy_type=PolicyType.TECH_INNOVATION_INCENTIVE,
            implementation_timeline={
                "2024": 0.5,  # 50% tax credit for qualifying investments
                "2025": 0.6,  # 60% tax credit
                "2026": 0.7,  # 70% tax credit
                "2027": 0.6,  # Scale back to 60%
                "2028": 0.5   # 50% long-term rate
            },
            geographic_scope=["USA", "EU", "China", "Japan", "South Korea", "Taiwan"],
            affected_sectors=["semiconductor_manufacturing", "equipment_supply", "R&D"],
            policy_stringency=0.4,  # Incentive, not mandate
            exemptions_and_flexibility={"all_eligible": 1.0}
        )
    
    def _initialize_baseline_economics(self):
        """Initialize baseline economic conditions for impact analysis."""
        
        # Major semiconductor companies baseline costs (annual, millions USD)
        self.baseline_economics = {
            "tsmc": {
                "annual_revenue": 75000,
                "energy_costs": 4500,
                "environmental_compliance": 800,
                "R&D_spending": 5600,
                "capital_expenditure": 40000,
                "operating_margin": 0.42
            },
            "samsung": {
                "annual_revenue": 65000,
                "energy_costs": 4000,
                "environmental_compliance": 750,
                "R&D_spending": 4800,
                "capital_expenditure": 35000,
                "operating_margin": 0.38
            },
            "intel": {
                "annual_revenue": 79000,
                "energy_costs": 3200,
                "environmental_compliance": 900,
                "R&D_spending": 15000,
                "capital_expenditure": 25000,
                "operating_margin": 0.25
            },
            "globalfoundries": {
                "annual_revenue": 6500,
                "energy_costs": 650,
                "environmental_compliance": 120,
                "R&D_spending": 400,
                "capital_expenditure": 2000,
                "operating_margin": 0.15
            },
            "smic": {
                "annual_revenue": 3570,
                "energy_costs": 480,
                "environmental_compliance": 45,
                "R&D_spending": 500,
                "capital_expenditure": 3000,
                "operating_margin": 0.20
            }
        }
    
    def _initialize_competitive_landscape(self):
        """Initialize competitive positioning baseline."""
        
        # Baseline competitive positions
        self.competitive_positions["tsmc"] = CompetitivePosition(
            company_id="tsmc",
            baseline_market_share=0.54,  # 54% foundry market share
            projected_market_share=0.54,
            cost_competitiveness_score=0.85,
            innovation_readiness_score=0.90,
            regulatory_compliance_score=0.75,
            overall_competitiveness_change=0.0
        )
        
        self.competitive_positions["samsung"] = CompetitivePosition(
            company_id="samsung",
            baseline_market_share=0.17,
            projected_market_share=0.17,
            cost_competitiveness_score=0.80,
            innovation_readiness_score=0.85,
            regulatory_compliance_score=0.70,
            overall_competitiveness_change=0.0
        )
        
        self.competitive_positions["intel"] = CompetitivePosition(
            company_id="intel",
            baseline_market_share=0.08,  # Foundry services
            projected_market_share=0.08,
            cost_competitiveness_score=0.65,
            innovation_readiness_score=0.95,
            regulatory_compliance_score=0.85,
            overall_competitiveness_change=0.0
        )
        
        self.competitive_positions["globalfoundries"] = CompetitivePosition(
            company_id="globalfoundries",
            baseline_market_share=0.06,
            projected_market_share=0.06,
            cost_competitiveness_score=0.70,
            innovation_readiness_score=0.60,
            regulatory_compliance_score=0.65,
            overall_competitiveness_change=0.0
        )
        
        self.competitive_positions["smic"] = CompetitivePosition(
            company_id="smic",
            baseline_market_share=0.05,
            projected_market_share=0.05,
            cost_competitiveness_score=0.90,  # Low-cost advantage
            innovation_readiness_score=0.50,
            regulatory_compliance_score=0.40,  # Weaker environmental compliance
            overall_competitiveness_change=0.0
        )
    
    def analyze_policy_impact(self, scenario_id: str, company_id: str, 
                            analysis_years: int = 5) -> Dict[str, Any]:
        """Analyze comprehensive economic impact of a policy scenario."""
        
        if scenario_id not in self.policy_scenarios:
            return {"error": "Policy scenario not found"}
        
        if company_id not in self.baseline_economics:
            return {"error": "Company not found"}
        
        scenario = self.policy_scenarios[scenario_id]
        baseline = self.baseline_economics[company_id]
        
        # Calculate impacts by category
        impacts = {}
        
        # Operational cost impacts
        impacts["operational_costs"] = self._calculate_operational_cost_impact(
            scenario, baseline, analysis_years)
        
        # Capital investment impacts
        impacts["capital_investments"] = self._calculate_capital_investment_impact(
            scenario, baseline, analysis_years)
        
        # Compliance cost impacts
        impacts["compliance_costs"] = self._calculate_compliance_cost_impact(
            scenario, baseline, analysis_years)
        
        # Revenue impacts
        impacts["revenue_impacts"] = self._calculate_revenue_impact(
            scenario, baseline, company_id, analysis_years)
        
        # Competitive positioning impacts
        impacts["competitive_impacts"] = self._calculate_competitive_impact(
            scenario, company_id, analysis_years)
        
        # Innovation and R&D impacts
        impacts["innovation_impacts"] = self._calculate_innovation_impact(
            scenario, baseline, analysis_years)
        
        # Calculate net present value
        impacts["financial_summary"] = self._calculate_financial_summary(
            impacts, analysis_years)
        
        # Risk and uncertainty analysis
        impacts["risk_analysis"] = self._analyze_implementation_risks(
            scenario, company_id)
        
        return {
            "scenario_id": scenario_id,
            "company_id": company_id,
            "analysis_period_years": analysis_years,
            "detailed_impacts": impacts,
            "summary_metrics": self._create_impact_summary(impacts),
            "policy_effectiveness": self._assess_policy_effectiveness(scenario, impacts)
        }
    
    def _calculate_operational_cost_impact(self, scenario: PolicyScenario, 
                                         baseline: Dict, years: int) -> Dict[str, Any]:
        """Calculate operational cost impacts."""
        
        annual_impacts = []
        
        for year in range(2024, 2024 + years):
            year_str = str(year)
            policy_intensity = scenario.implementation_timeline.get(year_str, 1.0)
            
            # Energy cost impacts
            if scenario.policy_type == PolicyType.CARBON_TAX:
                # Assume carbon intensity of 0.5 tons CO2 per $1000 energy cost
                carbon_cost_increase = baseline["energy_costs"] * 0.5 * policy_intensity * 250  # $250/ton max
                energy_efficiency_savings = baseline["energy_costs"] * policy_intensity * 0.15  # 15% efficiency gain
                net_energy_impact = carbon_cost_increase - energy_efficiency_savings
            
            elif scenario.policy_type == PolicyType.RENEWABLE_MANDATE:
                # Renewable energy premium costs
                renewable_premium = baseline["energy_costs"] * policy_intensity * 0.25  # 25% premium
                grid_stability_costs = baseline["energy_costs"] * policy_intensity * 0.1   # 10% for grid upgrades
                net_energy_impact = renewable_premium + grid_stability_costs
            
            elif scenario.policy_type == PolicyType.ENERGY_EFFICIENCY_STANDARD:
                # Efficiency requirements reduce energy costs
                efficiency_savings = baseline["energy_costs"] * policy_intensity
                maintenance_cost_increase = efficiency_savings * 0.3  # 30% of savings in maintenance
                net_energy_impact = -(efficiency_savings - maintenance_cost_increase)
            
            else:
                net_energy_impact = 0
            
            # Other operational impacts
            if scenario.policy_type == PolicyType.CIRCULAR_ECONOMY_REGULATION:
                waste_management_costs = baseline["annual_revenue"] * 0.005 * policy_intensity  # 0.5% of revenue
                material_recovery_savings = waste_management_costs * 0.4  # 40% savings from recovery
                net_operational_impact = waste_management_costs - material_recovery_savings
            else:
                net_operational_impact = 0
            
            total_operational_impact = net_energy_impact + net_operational_impact
            
            annual_impacts.append({
                "year": year,
                "energy_impact": net_energy_impact,
                "operational_impact": net_operational_impact,
                "total_impact": total_operational_impact,
                "policy_intensity": policy_intensity
            })
        
        total_impact = sum(impact["total_impact"] for impact in annual_impacts)
        
        return {
            "annual_breakdown": annual_impacts,
            "total_impact_millions": total_impact,
            "average_annual_impact": total_impact / years,
            "impact_as_percentage_of_revenue": (total_impact / years) / baseline["annual_revenue"] * 100
        }
    
    def _calculate_capital_investment_impact(self, scenario: PolicyScenario, 
                                           baseline: Dict, years: int) -> Dict[str, Any]:
        """Calculate capital investment impacts."""
        
        total_additional_capex = 0
        investment_breakdown = {}
        
        if scenario.policy_type == PolicyType.RENEWABLE_MANDATE:
            # On-site renewable generation investment
            renewable_investment = baseline["annual_revenue"] * 0.08  # 8% of revenue
            total_additional_capex += renewable_investment
            investment_breakdown["renewable_generation"] = renewable_investment
            
        elif scenario.policy_type == PolicyType.ENERGY_EFFICIENCY_STANDARD:
            # Energy efficiency equipment upgrades
            efficiency_investment = baseline["capital_expenditure"] * 0.15  # 15% additional capex
            total_additional_capex += efficiency_investment
            investment_breakdown["efficiency_upgrades"] = efficiency_investment
            
        elif scenario.policy_type == PolicyType.CIRCULAR_ECONOMY_REGULATION:
            # Recycling and waste management infrastructure
            circular_investment = baseline["annual_revenue"] * 0.03  # 3% of revenue
            total_additional_capex += circular_investment
            investment_breakdown["circular_infrastructure"] = circular_investment
        
        if scenario.policy_type == PolicyType.TECH_INNOVATION_INCENTIVE:
            # Additional R&D investment stimulated by incentives
            incentive_multiplier = max(scenario.implementation_timeline.values())
            additional_rd = baseline["R&D_spending"] * 0.2 * incentive_multiplier  # 20% increase
            # This is partially offset by tax credits
            net_rd_cost = additional_rd * (1 - incentive_multiplier)
            total_additional_capex += net_rd_cost
            investment_breakdown["innovation_investment"] = additional_rd
            investment_breakdown["tax_credit_offset"] = -additional_rd * incentive_multiplier
        
        # Calculate payback and returns
        if total_additional_capex > 0:
            # Estimate annual savings/benefits
            if scenario.policy_type == PolicyType.ENERGY_EFFICIENCY_STANDARD:
                annual_savings = baseline["energy_costs"] * 0.2  # 20% energy savings
            elif scenario.policy_type == PolicyType.RENEWABLE_MANDATE:
                annual_savings = baseline["energy_costs"] * 0.1  # 10% long-term savings
            elif scenario.policy_type == PolicyType.CIRCULAR_ECONOMY_REGULATION:
                annual_savings = baseline["annual_revenue"] * 0.01  # 1% material cost savings
            else:
                annual_savings = total_additional_capex * 0.08  # 8% generic return
            
            payback_period = total_additional_capex / annual_savings if annual_savings > 0 else float('inf')
        else:
            annual_savings = 0
            payback_period = 0
        
        return {
            "total_additional_capex_millions": total_additional_capex,
            "investment_breakdown": investment_breakdown,
            "annual_savings_millions": annual_savings,
            "payback_period_years": payback_period,
            "capex_increase_percentage": total_additional_capex / baseline["capital_expenditure"] * 100
        }
    
    def _calculate_compliance_cost_impact(self, scenario: PolicyScenario, 
                                        baseline: Dict, years: int) -> Dict[str, Any]:
        """Calculate regulatory compliance cost impacts."""
        
        # Base compliance cost increase by policy type
        if scenario.policy_type == PolicyType.CARBON_TAX:
            base_compliance_cost = baseline["environmental_compliance"] * 0.3  # 30% increase
        elif scenario.policy_type == PolicyType.EMISSIONS_TRADING:
            base_compliance_cost = baseline["environmental_compliance"] * 0.4  # 40% increase
        elif scenario.policy_type == PolicyType.CIRCULAR_ECONOMY_REGULATION:
            base_compliance_cost = baseline["environmental_compliance"] * 0.6  # 60% increase
        elif scenario.policy_type == PolicyType.ENERGY_EFFICIENCY_STANDARD:
            base_compliance_cost = baseline["environmental_compliance"] * 0.2  # 20% increase
        else:
            base_compliance_cost = baseline["environmental_compliance"] * 0.1  # 10% increase
        
        # Adjust for policy stringency
        adjusted_compliance_cost = base_compliance_cost * scenario.policy_stringency
        
        # One-time implementation costs
        implementation_cost = adjusted_compliance_cost * 2  # 2x annual cost for setup
        
        # Annual ongoing costs
        annual_compliance_cost = adjusted_compliance_cost
        
        total_cost = implementation_cost + (annual_compliance_cost * years)
        
        return {
            "implementation_cost_millions": implementation_cost,
            "annual_compliance_cost_millions": annual_compliance_cost,
            "total_compliance_cost_millions": total_cost,
            "compliance_cost_as_percentage_of_revenue": annual_compliance_cost / baseline["annual_revenue"] * 100
        }
    
    def _calculate_revenue_impact(self, scenario: PolicyScenario, baseline: Dict, 
                                company_id: str, years: int) -> Dict[str, Any]:
        """Calculate revenue impacts from policy implementation."""
        
        # Market demand effects
        if scenario.policy_type in [PolicyType.CARBON_TAX, PolicyType.EMISSIONS_TRADING]:
            # Increased demand for energy-efficient chips
            demand_boost = 0.05  # 5% increase in demand
        elif scenario.policy_type == PolicyType.TECH_INNOVATION_INCENTIVE:
            # Innovation incentives boost advanced chip demand
            demand_boost = 0.08  # 8% increase
        else:
            demand_boost = 0.02  # 2% general boost
        
        # Premium pricing for green products
        if company_id in ["intel", "tsmc"]:  # Technology leaders
            green_premium = 0.03  # 3% premium for green products
        else:
            green_premium = 0.01  # 1% premium for others
        
        # Market share effects based on compliance readiness
        competitive_position = self.competitive_positions[company_id]
        if competitive_position.regulatory_compliance_score > 0.7:
            market_share_gain = 0.02  # 2% market share gain
        elif competitive_position.regulatory_compliance_score < 0.5:
            market_share_gain = -0.03  # 3% market share loss
        else:
            market_share_gain = 0
        
        # Calculate revenue impacts
        annual_revenue_impact = baseline["annual_revenue"] * (
            demand_boost + green_premium + market_share_gain
        )
        
        total_revenue_impact = annual_revenue_impact * years
        
        return {
            "annual_revenue_impact_millions": annual_revenue_impact,
            "total_revenue_impact_millions": total_revenue_impact,
            "demand_boost_percentage": demand_boost * 100,
            "green_premium_percentage": green_premium * 100,
            "market_share_change_percentage": market_share_gain * 100,
            "revenue_impact_breakdown": {
                "demand_effect": baseline["annual_revenue"] * demand_boost,
                "premium_effect": baseline["annual_revenue"] * green_premium,
                "market_share_effect": baseline["annual_revenue"] * market_share_gain
            }
        }
    
    def _calculate_competitive_impact(self, scenario: PolicyScenario, 
                                    company_id: str, years: int) -> Dict[str, Any]:
        """Calculate competitive positioning impacts."""
        
        position = self.competitive_positions[company_id].copy()
        
        # Update scores based on policy type
        if scenario.policy_type == PolicyType.TECH_INNOVATION_INCENTIVE:
            # Innovation incentives favor companies with strong R&D
            if position.innovation_readiness_score > 0.8:
                position.cost_competitiveness_score += 0.05
                position.innovation_readiness_score += 0.03
        
        elif scenario.policy_type in [PolicyType.CARBON_TAX, PolicyType.EMISSIONS_TRADING]:
            # Carbon policies favor efficient, compliant companies
            if position.regulatory_compliance_score > 0.7:
                position.cost_competitiveness_score += 0.03
            else:
                position.cost_competitiveness_score -= 0.05
        
        elif scenario.policy_type == PolicyType.RENEWABLE_MANDATE:
            # Renewable mandates affect companies differently by location
            if company_id in ["intel"]:  # US companies with renewable access
                position.cost_competitiveness_score += 0.02
            elif company_id in ["smic"]:  # Companies in coal-heavy grids
                position.cost_competitiveness_score -= 0.04
        
        # Calculate overall competitiveness change
        baseline_score = (position.cost_competitiveness_score + 
                         position.innovation_readiness_score + 
                         position.regulatory_compliance_score) / 3
        
        current_score = (position.cost_competitiveness_score + 
                        position.innovation_readiness_score + 
                        position.regulatory_compliance_score) / 3
        
        competitiveness_change = (current_score - baseline_score) / baseline_score * 100
        
        # Project market share changes
        market_share_elasticity = 0.5  # 50% of competitiveness change translates to market share
        projected_share_change = competitiveness_change * market_share_elasticity / 100
        new_market_share = position.baseline_market_share * (1 + projected_share_change)
        
        return {
            "baseline_competitiveness_score": baseline_score,
            "projected_competitiveness_score": current_score,
            "competitiveness_change_percentage": competitiveness_change,
            "baseline_market_share": position.baseline_market_share,
            "projected_market_share": new_market_share,
            "market_share_change_percentage": projected_share_change * 100,
            "score_breakdown": {
                "cost_competitiveness": position.cost_competitiveness_score,
                "innovation_readiness": position.innovation_readiness_score,
                "regulatory_compliance": position.regulatory_compliance_score
            }
        }
    
    def _calculate_innovation_impact(self, scenario: PolicyScenario, 
                                   baseline: Dict, years: int) -> Dict[str, Any]:
        """Calculate innovation and R&D impacts."""
        
        if scenario.policy_type == PolicyType.TECH_INNOVATION_INCENTIVE:
            # Direct innovation stimulus
            rd_increase_percentage = max(scenario.implementation_timeline.values()) * 30  # Up to 30% increase
            additional_rd = baseline["R&D_spending"] * rd_increase_percentage / 100
            
            # Innovation productivity boost
            innovation_productivity_gain = 0.15  # 15% higher productivity from targeted incentives
            
        elif scenario.policy_type in [PolicyType.ENERGY_EFFICIENCY_STANDARD, 
                                     PolicyType.RENEWABLE_MANDATE]:
            # Indirect innovation stimulus from compliance needs
            rd_increase_percentage = 8  # 8% increase
            additional_rd = baseline["R&D_spending"] * 0.08
            innovation_productivity_gain = 0.05  # 5% gain
            
        else:
            rd_increase_percentage = 2  # 2% general increase
            additional_rd = baseline["R&D_spending"] * 0.02
            innovation_productivity_gain = 0.02  # 2% gain
        
        # Patent and IP value impacts
        if scenario.policy_type == PolicyType.TECH_INNOVATION_INCENTIVE:
            patent_value_increase = 0.2  # 20% increase in patent value
        else:
            patent_value_increase = 0.05  # 5% general increase
        
        # Technology leadership impacts
        current_rd_intensity = baseline["R&D_spending"] / baseline["annual_revenue"]
        new_rd_intensity = (baseline["R&D_spending"] + additional_rd) / baseline["annual_revenue"]
        
        return {
            "rd_increase_percentage": rd_increase_percentage,
            "additional_rd_spending_millions": additional_rd,
            "innovation_productivity_gain_percentage": innovation_productivity_gain * 100,
            "patent_value_increase_percentage": patent_value_increase * 100,
            "baseline_rd_intensity": current_rd_intensity,
            "projected_rd_intensity": new_rd_intensity,
            "technology_leadership_score_change": innovation_productivity_gain * 10  # 0-1 scale
        }
    
    def _calculate_financial_summary(self, impacts: Dict, years: int) -> Dict[str, Any]:
        """Calculate comprehensive financial summary."""
        
        # Annual cash flows
        annual_costs = (
            impacts["operational_costs"]["average_annual_impact"] +
            impacts["compliance_costs"]["annual_compliance_cost_millions"] +
            impacts["capital_investments"]["total_additional_capex_millions"] / years
        )
        
        annual_benefits = (
            impacts["revenue_impacts"]["annual_revenue_impact_millions"] +
            impacts["capital_investments"]["annual_savings_millions"] +
            impacts["innovation_impacts"]["additional_rd_spending_millions"] * 0.3  # 30% ROI on R&D
        )
        
        net_annual_impact = annual_benefits - annual_costs
        
        # NPV calculation (using 8% discount rate)
        discount_rate = 0.08
        npv = sum(net_annual_impact / (1 + discount_rate) ** year for year in range(1, years + 1))
        
        # ROI calculation
        total_investment = (
            impacts["compliance_costs"]["implementation_cost_millions"] +
            impacts["capital_investments"]["total_additional_capex_millions"]
        )
        
        if total_investment > 0:
            roi = (npv / total_investment) * 100
        else:
            roi = float('inf') if npv > 0 else 0
        
        return {
            "annual_costs_millions": annual_costs,
            "annual_benefits_millions": annual_benefits,
            "net_annual_impact_millions": net_annual_impact,
            "net_present_value_millions": npv,
            "total_investment_millions": total_investment,
            "roi_percentage": roi,
            "payback_period_years": total_investment / net_annual_impact if net_annual_impact > 0 else float('inf'),
            "break_even_year": total_investment / net_annual_impact if net_annual_impact > 0 else None
        }
    
    def _analyze_implementation_risks(self, scenario: PolicyScenario, 
                                    company_id: str) -> Dict[str, Any]:
        """Analyze implementation risks and uncertainties."""
        
        # Policy implementation risks
        policy_risk_factors = {
            "regulatory_uncertainty": 0.3,  # 30% uncertainty in final regulations
            "timeline_delays": 0.2,         # 20% risk of implementation delays
            "exemption_changes": 0.15,      # 15% risk of exemption modifications
            "international_coordination": 0.25  # 25% risk from lack of global coordination
        }
        
        # Company-specific risks
        company_position = self.competitive_positions[company_id]
        
        if company_position.regulatory_compliance_score < 0.5:
            compliance_risk = 0.4  # High risk for non-compliant companies
        elif company_position.regulatory_compliance_score > 0.8:
            compliance_risk = 0.1  # Low risk for compliant companies
        else:
            compliance_risk = 0.25  # Medium risk
        
        if company_position.innovation_readiness_score < 0.6:
            technology_risk = 0.35  # High technology adaptation risk
        else:
            technology_risk = 0.15  # Low technology risk
        
        # Market risks
        market_risks = {
            "demand_volatility": 0.2,       # 20% demand uncertainty
            "competitive_response": 0.25,   # 25% uncertainty in competitor actions
            "cost_escalation": 0.3,         # 30% risk of higher than expected costs
            "technology_disruption": 0.2    # 20% risk of disruptive technologies
        }
        
        # Overall risk assessment
        overall_risk_score = (
            np.mean(list(policy_risk_factors.values())) * 0.4 +
            compliance_risk * 0.3 +
            technology_risk * 0.2 +
            np.mean(list(market_risks.values())) * 0.1
        )
        
        return {
            "overall_risk_score": overall_risk_score,
            "risk_level": "High" if overall_risk_score > 0.6 else "Medium" if overall_risk_score > 0.3 else "Low",
            "policy_risks": policy_risk_factors,
            "company_risks": {
                "compliance_risk": compliance_risk,
                "technology_risk": technology_risk
            },
            "market_risks": market_risks,
            "mitigation_strategies": self._suggest_risk_mitigation(scenario, company_id, overall_risk_score)
        }
    
    def _suggest_risk_mitigation(self, scenario: PolicyScenario, 
                               company_id: str, risk_score: float) -> List[str]:
        """Suggest risk mitigation strategies."""
        
        strategies = []
        
        if risk_score > 0.5:
            strategies.append("Develop comprehensive compliance monitoring system")
            strategies.append("Establish regulatory affairs team with policy expertise")
            strategies.append("Create contingency plans for multiple policy scenarios")
        
        if scenario.policy_type == PolicyType.CARBON_TAX:
            strategies.append("Invest in carbon management and tracking systems")
            strategies.append("Explore carbon offset partnerships")
            strategies.append("Accelerate energy efficiency programs")
        
        if scenario.policy_type == PolicyType.RENEWABLE_MANDATE:
            strategies.append("Secure long-term renewable energy contracts")
            strategies.append("Invest in on-site renewable generation")
            strategies.append("Partner with utility companies for grid upgrades")
        
        if scenario.policy_type == PolicyType.TECH_INNOVATION_INCENTIVE:
            strategies.append("Establish dedicated green technology R&D programs")
            strategies.append("Form research partnerships with universities")
            strategies.append("Create IP portfolio for environmental technologies")
        
        company_position = self.competitive_positions[company_id]
        if company_position.regulatory_compliance_score < 0.6:
            strategies.append("Hire environmental compliance experts")
            strategies.append("Implement environmental management systems")
            strategies.append("Conduct regular compliance audits")
        
        return strategies
    
    def _create_impact_summary(self, impacts: Dict) -> Dict[str, Any]:
        """Create high-level impact summary."""
        
        return {
            "net_financial_impact_millions": impacts["financial_summary"]["net_present_value_millions"],
            "roi_percentage": impacts["financial_summary"]["roi_percentage"],
            "competitiveness_change": impacts["competitive_impacts"]["competitiveness_change_percentage"],
            "market_share_impact": impacts["competitive_impacts"]["market_share_change_percentage"],
            "innovation_boost": impacts["innovation_impacts"]["rd_increase_percentage"],
            "implementation_risk": impacts["risk_analysis"]["overall_risk_score"],
            "overall_assessment": self._overall_impact_assessment(impacts)
        }
    
    def _overall_impact_assessment(self, impacts: Dict) -> str:
        """Provide overall assessment of policy impact."""
        
        npv = impacts["financial_summary"]["net_present_value_millions"]
        roi = impacts["financial_summary"]["roi_percentage"]
        competitiveness = impacts["competitive_impacts"]["competitiveness_change_percentage"]
        risk = impacts["risk_analysis"]["overall_risk_score"]
        
        if npv > 0 and roi > 15 and competitiveness > 0 and risk < 0.4:
            return "Highly Positive - Strong financial returns and competitive advantage"
        elif npv > 0 and roi > 8:
            return "Positive - Good financial returns despite some challenges"
        elif npv > -100 and competitiveness > -5:
            return "Mixed - Some benefits offset by costs and risks"
        elif npv < 0 and competitiveness < -5:
            return "Negative - Significant costs and competitive disadvantage"
        else:
            return "Challenging - High costs and risks with uncertain benefits"
    
    def _assess_policy_effectiveness(self, scenario: PolicyScenario, 
                                   impacts: Dict) -> Dict[str, Any]:
        """Assess overall policy effectiveness."""
        
        # Environmental effectiveness (assumed based on policy type)
        if scenario.policy_type == PolicyType.CARBON_TAX:
            environmental_effectiveness = 0.8  # High effectiveness for carbon reduction
        elif scenario.policy_type == PolicyType.RENEWABLE_MANDATE:
            environmental_effectiveness = 0.7  # Good for renewable adoption
        elif scenario.policy_type == PolicyType.ENERGY_EFFICIENCY_STANDARD:
            environmental_effectiveness = 0.6  # Moderate for overall emissions
        else:
            environmental_effectiveness = 0.5  # Moderate effectiveness
        
        # Economic efficiency
        roi = impacts["financial_summary"]["roi_percentage"]
        if roi > 20:
            economic_efficiency = 0.9
        elif roi > 10:
            economic_efficiency = 0.7
        elif roi > 0:
            economic_efficiency = 0.5
        else:
            economic_efficiency = 0.3
        
        # Innovation stimulus
        rd_increase = impacts["innovation_impacts"]["rd_increase_percentage"]
        innovation_effectiveness = min(1.0, rd_increase / 20)  # Max effectiveness at 20% increase
        
        # Overall policy score
        overall_score = (environmental_effectiveness * 0.4 + 
                        economic_efficiency * 0.4 + 
                        innovation_effectiveness * 0.2)
        
        return {
            "environmental_effectiveness": environmental_effectiveness,
            "economic_efficiency": economic_efficiency,
            "innovation_effectiveness": innovation_effectiveness,
            "overall_policy_score": overall_score,
            "policy_rating": "Excellent" if overall_score > 0.8 else 
                           "Good" if overall_score > 0.6 else 
                           "Fair" if overall_score > 0.4 else "Poor",
            "key_strengths": self._identify_policy_strengths(scenario, impacts),
            "improvement_recommendations": self._suggest_policy_improvements(scenario, impacts)
        }
    
    def _identify_policy_strengths(self, scenario: PolicyScenario, impacts: Dict) -> List[str]:
        """Identify key policy strengths."""
        
        strengths = []
        
        if impacts["financial_summary"]["roi_percentage"] > 15:
            strengths.append("Strong positive financial returns")
        
        if impacts["innovation_impacts"]["rd_increase_percentage"] > 10:
            strengths.append("Significant innovation stimulus")
        
        if impacts["competitive_impacts"]["competitiveness_change_percentage"] > 5:
            strengths.append("Competitive advantage for early adopters")
        
        if impacts["risk_analysis"]["overall_risk_score"] < 0.3:
            strengths.append("Low implementation risk")
        
        if scenario.policy_stringency > 0.7:
            strengths.append("Clear environmental objectives")
        
        return strengths
    
    def _suggest_policy_improvements(self, scenario: PolicyScenario, impacts: Dict) -> List[str]:
        """Suggest policy design improvements."""
        
        improvements = []
        
        if impacts["financial_summary"]["roi_percentage"] < 5:
            improvements.append("Increase financial incentives or reduce compliance costs")
        
        if impacts["risk_analysis"]["overall_risk_score"] > 0.6:
            improvements.append("Provide clearer regulatory guidance and longer transition periods")
        
        if impacts["competitive_impacts"]["market_share_change_percentage"] < -5:
            improvements.append("Add support for companies in competitive disadvantage")
        
        if impacts["innovation_impacts"]["rd_increase_percentage"] < 5:
            improvements.append("Strengthen innovation incentives and R&D support")
        
        if len(scenario.exemptions_and_flexibility) < 2:
            improvements.append("Consider additional flexibility mechanisms for diverse industry needs")
        
        return improvements
    
    def get_economic_impact_summary(self) -> Dict[str, Any]:
        """Get comprehensive economic impact summary across all scenarios."""
        
        # This would aggregate results across multiple scenarios and companies
        # For now, return a framework summary
        
        return {
            "policy_scenarios_analyzed": len(self.policy_scenarios),
            "companies_in_scope": len(self.baseline_economics),
            "policy_types_covered": list(set(scenario.policy_type.value for scenario in self.policy_scenarios.values())),
            "geographic_coverage": list(set().union(*[scenario.geographic_scope for scenario in self.policy_scenarios.values()])),
            "analysis_capabilities": [
                "Multi-stakeholder impact assessment",
                "Competitive positioning analysis", 
                "Cost-benefit analysis with uncertainty",
                "Innovation and R&D impact modeling",
                "Risk assessment and mitigation planning",
                "Policy effectiveness evaluation",
                "Financial modeling with NPV and ROI",
                "Macroeconomic effect analysis"
            ],
            "key_insights": [
                "Innovation incentives show highest ROI potential",
                "Carbon pricing creates competitive advantages for efficient companies",
                "Renewable mandates have varying impacts based on regional grid mix",
                "Circular economy regulations require significant upfront investment",
                "Policy coordination across regions critical for effectiveness"
            ]
        } 