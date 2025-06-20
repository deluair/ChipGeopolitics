"""
Strategic Competition Model for Semiconductor Industry

Models strategic competition dynamics between major powers including technology rivalry,
R&D investment strategies, innovation races, and competitive positioning in critical technologies.
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

class CompetitionDomain(Enum):
    """Strategic competition domains."""
    ADVANCED_SEMICONDUCTORS = "advanced_semiconductors"
    MANUFACTURING_EQUIPMENT = "manufacturing_equipment"
    DESIGN_SOFTWARE = "design_software"
    MATERIALS_CHEMICALS = "materials_chemicals"
    PACKAGING_ASSEMBLY = "packaging_assembly"
    AI_ACCELERATORS = "ai_accelerators"
    QUANTUM_COMPUTING = "quantum_computing"
    MEMORY_TECHNOLOGIES = "memory_technologies"

class CompetitiveStrategy(Enum):
    """Strategic approaches to competition."""
    INNOVATION_LEADERSHIP = "innovation_leadership"
    COST_COMPETITION = "cost_competition"
    VERTICAL_INTEGRATION = "vertical_integration"
    ECOSYSTEM_BUILDING = "ecosystem_building"
    EXPORT_RESTRICTIONS = "export_restrictions"
    INDIGENOUS_DEVELOPMENT = "indigenous_development"
    TECHNOLOGY_ACQUISITION = "technology_acquisition"
    STANDARDS_SETTING = "standards_setting"

@dataclass
class NationalCapability:
    """National capability in a competition domain."""
    country: str
    domain: CompetitionDomain
    current_capability: float  # 0-1 capability score
    r_and_d_investment_billions: float
    market_share: float
    key_players: List[str]  # Major companies/institutions
    technology_strengths: List[str]
    weaknesses: List[str]
    strategic_priorities: List[CompetitiveStrategy]
    government_support_score: float  # 0-1 support level

@dataclass
class CompetitiveAction:
    """Strategic competitive action."""
    action_id: str
    actor_country: str
    target_countries: List[str]
    domain: CompetitionDomain
    action_type: CompetitiveStrategy
    investment_billions: float
    timeline_years: int
    expected_impact: float  # 0-1 impact score
    countermeasure_probability: float
    implementation_date: datetime

@dataclass
class InnovationRace:
    """Technology innovation race analysis."""
    race_id: str
    domain: CompetitionDomain
    participating_countries: List[str]
    technology_target: str
    current_leaders: List[str]
    capability_gaps: Dict[str, float]  # Country -> gap from leader
    investment_levels: Dict[str, float]  # Country -> annual investment
    projected_outcomes: Dict[str, Dict[str, float]]  # Country -> scenario probabilities
    breakthrough_timeline: Dict[str, int]  # Country -> years to breakthrough

class StrategicCompetitionModel:
    """
    Comprehensive strategic competition modeling for semiconductor industry.
    
    Models:
    - Multi-domain competition analysis
    - R&D investment effectiveness
    - Innovation race dynamics
    - Competitive positioning changes
    - Strategic action-reaction cycles
    - Long-term competitive landscapes
    """
    
    def __init__(self):
        # Competition state
        self.national_capabilities: Dict[str, Dict[CompetitionDomain, NationalCapability]] = {}
        self.competitive_actions: List[CompetitiveAction] = []
        self.innovation_races: Dict[str, InnovationRace] = {}
        
        # Analysis results
        self.competition_dynamics: Dict[str, Any] = {}
        self.strategic_forecasts: Dict[str, Dict[str, float]] = {}
        self.policy_scenarios: List[Dict[str, Any]] = []
        
        # Initialize with realistic 2024 competition landscape
        self._initialize_national_capabilities()
        self._initialize_innovation_races()
    
    def _initialize_national_capabilities(self):
        """Initialize realistic national capabilities by domain."""
        
        # USA capabilities
        usa_capabilities = {
            CompetitionDomain.ADVANCED_SEMICONDUCTORS: NationalCapability(
                country="USA",
                domain=CompetitionDomain.ADVANCED_SEMICONDUCTORS,
                current_capability=0.85,
                r_and_d_investment_billions=50.0,
                market_share=0.45,
                key_players=["Intel", "NVIDIA", "AMD", "Qualcomm", "Broadcom"],
                technology_strengths=["CPU design", "GPU architecture", "AI accelerators", "wireless"],
                weaknesses=["manufacturing capacity", "cost competitiveness"],
                strategic_priorities=[CompetitiveStrategy.INNOVATION_LEADERSHIP, CompetitiveStrategy.EXPORT_RESTRICTIONS],
                government_support_score=0.8
            ),
            CompetitionDomain.MANUFACTURING_EQUIPMENT: NationalCapability(
                country="USA",
                domain=CompetitionDomain.MANUFACTURING_EQUIPMENT,
                current_capability=0.75,
                r_and_d_investment_billions=15.0,
                market_share=0.35,
                key_players=["Applied Materials", "Lam Research", "KLA"],
                technology_strengths=["deposition", "etch", "metrology"],
                weaknesses=["lithography", "cost pressure"],
                strategic_priorities=[CompetitiveStrategy.INNOVATION_LEADERSHIP, CompetitiveStrategy.EXPORT_RESTRICTIONS],
                government_support_score=0.75
            ),
            CompetitionDomain.DESIGN_SOFTWARE: NationalCapability(
                country="USA",
                domain=CompetitionDomain.DESIGN_SOFTWARE,
                current_capability=0.95,
                r_and_d_investment_billions=8.0,
                market_share=0.65,
                key_players=["Synopsys", "Cadence", "Mentor Graphics"],
                technology_strengths=["EDA tools", "verification", "AI-assisted design"],
                weaknesses=["cost for emerging markets"],
                strategic_priorities=[CompetitiveStrategy.INNOVATION_LEADERSHIP, CompetitiveStrategy.ECOSYSTEM_BUILDING],
                government_support_score=0.7
            )
        }
        
        # China capabilities
        china_capabilities = {
            CompetitionDomain.ADVANCED_SEMICONDUCTORS: NationalCapability(
                country="China",
                domain=CompetitionDomain.ADVANCED_SEMICONDUCTORS,
                current_capability=0.6,
                r_and_d_investment_billions=75.0,
                market_share=0.15,
                key_players=["SMIC", "Loongson", "HiSilicon", "UNISOC"],
                technology_strengths=["mature nodes", "system integration", "cost efficiency"],
                weaknesses=["advanced processes", "design capabilities", "materials"],
                strategic_priorities=[CompetitiveStrategy.INDIGENOUS_DEVELOPMENT, CompetitiveStrategy.VERTICAL_INTEGRATION],
                government_support_score=0.95
            ),
            CompetitionDomain.MANUFACTURING_EQUIPMENT: NationalCapability(
                country="China",
                domain=CompetitionDomain.MANUFACTURING_EQUIPMENT,
                current_capability=0.35,
                r_and_d_investment_billions=25.0,
                market_share=0.08,
                key_players=["NAURA", "ACM Research", "AMEC"],
                technology_strengths=["packaging equipment", "mature process tools"],
                weaknesses=["advanced lithography", "precision equipment", "materials"],
                strategic_priorities=[CompetitiveStrategy.INDIGENOUS_DEVELOPMENT, CompetitiveStrategy.TECHNOLOGY_ACQUISITION],
                government_support_score=0.98
            )
        }
        
        # Store additional country capabilities
        taiwan_capabilities = {
            CompetitionDomain.ADVANCED_SEMICONDUCTORS: NationalCapability(
                country="Taiwan", domain=CompetitionDomain.ADVANCED_SEMICONDUCTORS,
                current_capability=0.95, r_and_d_investment_billions=20.0, market_share=0.25,
                key_players=["TSMC", "MediaTek", "ASE Group", "UMC"],
                technology_strengths=["foundry services", "advanced packaging", "process technology"],
                weaknesses=["design capabilities", "materials", "equipment"],
                strategic_priorities=[CompetitiveStrategy.INNOVATION_LEADERSHIP, CompetitiveStrategy.ECOSYSTEM_BUILDING],
                government_support_score=0.85
            )
        }
        
        korea_capabilities = {
            CompetitionDomain.MEMORY_TECHNOLOGIES: NationalCapability(
                country="South Korea", domain=CompetitionDomain.MEMORY_TECHNOLOGIES,
                current_capability=0.9, r_and_d_investment_billions=18.0, market_share=0.45,
                key_players=["Samsung", "SK Hynix"],
                technology_strengths=["DRAM", "NAND flash", "next-gen memory"],
                weaknesses=["logic processors", "design tools"],
                strategic_priorities=[CompetitiveStrategy.INNOVATION_LEADERSHIP, CompetitiveStrategy.VERTICAL_INTEGRATION],
                government_support_score=0.8
            )
        }
        
        # Store capabilities
        self.national_capabilities = {
            "USA": usa_capabilities,
            "China": china_capabilities,
            "Taiwan": taiwan_capabilities,
            "South Korea": korea_capabilities
        }
    
    def _initialize_innovation_races(self):
        """Initialize ongoing innovation races."""
        
        # Advanced node race (sub-3nm)
        self.innovation_races["advanced_nodes"] = InnovationRace(
            race_id="advanced_nodes",
            domain=CompetitionDomain.ADVANCED_SEMICONDUCTORS,
            participating_countries=["USA", "Taiwan", "South Korea", "China"],
            technology_target="Sub-2nm process technology",
            current_leaders=["Taiwan", "South Korea"],
            capability_gaps={
                "Taiwan": 0.0,  # Leader
                "South Korea": 0.1,
                "USA": 0.3,
                "China": 0.6
            },
            investment_levels={
                "Taiwan": 15.0,
                "South Korea": 12.0,
                "USA": 8.0,
                "China": 20.0
            },
            projected_outcomes={
                "Taiwan": {"maintains_lead": 0.6, "loses_lead": 0.4},
                "South Korea": {"takes_lead": 0.3, "maintains_position": 0.5, "falls_behind": 0.2},
                "USA": {"catches_up": 0.3, "maintains_gap": 0.5, "falls_further": 0.2},
                "China": {"catches_up": 0.4, "maintains_gap": 0.4, "falls_further": 0.2}
            },
            breakthrough_timeline={
                "Taiwan": 2,
                "South Korea": 2,
                "USA": 4,
                "China": 6
            }
        )
        
        # AI accelerator race
        self.innovation_races["ai_accelerators"] = InnovationRace(
            race_id="ai_accelerators",
            domain=CompetitionDomain.AI_ACCELERATORS,
            participating_countries=["USA", "China", "Taiwan"],
            technology_target="Next-generation AI training chips",
            current_leaders=["USA"],
            capability_gaps={
                "USA": 0.0,
                "Taiwan": 0.2,
                "China": 0.4
            },
            investment_levels={
                "USA": 25.0,
                "China": 30.0,
                "Taiwan": 8.0
            },
            projected_outcomes={
                "USA": {"maintains_lead": 0.7, "loses_lead": 0.3},
                "China": {"takes_lead": 0.4, "closes_gap": 0.4, "maintains_gap": 0.2},
                "Taiwan": {"closes_gap": 0.3, "maintains_gap": 0.5, "falls_behind": 0.2}
            },
            breakthrough_timeline={
                "USA": 1,
                "China": 3,
                "Taiwan": 4
            }
        )
    
    def analyze_competitive_dynamics(self, time_horizon_years: int = 10) -> Dict[str, Any]:
        """Analyze competitive dynamics across all domains."""
        
        dynamics_analysis = {
            "current_leaders_by_domain": {},
            "investment_trends": {},
            "capability_trajectories": {},
            "competitive_pressure_points": [],
            "strategic_vulnerabilities": {},
            "power_shifts": {}
        }
        
        # Identify current leaders by domain
        for domain in CompetitionDomain:
            domain_capabilities = []
            for country, capabilities in self.national_capabilities.items():
                if domain in capabilities:
                    capability = capabilities[domain]
                    domain_capabilities.append((country, capability.current_capability))
            
            if domain_capabilities:
                domain_capabilities.sort(key=lambda x: x[1], reverse=True)
                dynamics_analysis["current_leaders_by_domain"][domain.value] = domain_capabilities[:3]
        
        # Investment trend analysis
        total_investments = {}
        for country, capabilities in self.national_capabilities.items():
            total_investment = sum(cap.r_and_d_investment_billions for cap in capabilities.values())
            total_investments[country] = total_investment
        
        dynamics_analysis["investment_trends"] = {
            "total_by_country": total_investments,
            "investment_intensity": self._calculate_investment_intensity(),
            "strategic_focus_areas": self._identify_strategic_focus_areas()
        }
        
        # Capability trajectory projections
        dynamics_analysis["capability_trajectories"] = self._project_capability_trajectories(time_horizon_years)
        
        # Store analysis
        self.competition_dynamics = dynamics_analysis
        return dynamics_analysis
    
    def _calculate_investment_intensity(self) -> Dict[str, float]:
        """Calculate R&D investment intensity by country."""
        # Simplified GDP estimates for intensity calculation
        gdp_estimates = {
            "USA": 25000,  # $25T
            "China": 18000,  # $18T
            "Taiwan": 800,   # $800B
            "South Korea": 2000,  # $2T
        }
        
        intensity = {}
        for country in self.national_capabilities.keys():
            total_investment = sum(cap.r_and_d_investment_billions 
                                 for cap in self.national_capabilities[country].values())
            gdp = gdp_estimates.get(country, 1000)
            intensity[country] = (total_investment / gdp) * 100  # Percentage of GDP
        
        return intensity
    
    def _identify_strategic_focus_areas(self) -> Dict[str, List[CompetitionDomain]]:
        """Identify strategic focus areas by country based on investment allocation."""
        focus_areas = {}
        
        for country, capabilities in self.national_capabilities.items():
            # Sort domains by investment level
            domain_investments = [(domain, cap.r_and_d_investment_billions) 
                                for domain, cap in capabilities.items()]
            domain_investments.sort(key=lambda x: x[1], reverse=True)
            
            # Top 2-3 domains are strategic focus areas
            focus_areas[country] = [domain for domain, _ in domain_investments[:3]]
        
        return focus_areas
    
    def _project_capability_trajectories(self, years: int) -> Dict[str, Dict[str, List[float]]]:
        """Project capability development trajectories."""
        trajectories = {}
        
        for country, capabilities in self.national_capabilities.items():
            country_trajectories = {}
            
            for domain, capability in capabilities.items():
                # Simple model: capability growth based on investment and current level
                current_cap = capability.current_capability
                investment = capability.r_and_d_investment_billions
                gov_support = capability.government_support_score
                
                # Growth rate based on investment intensity and government support
                base_growth_rate = (investment / 10) * 0.02 * gov_support  # Max ~10% per year
                
                # Diminishing returns as capability approaches 1.0
                trajectory = [current_cap]
                for year in range(1, years + 1):
                    growth_rate = base_growth_rate * (1 - trajectory[-1])  # Diminishing returns
                    new_capability = min(1.0, trajectory[-1] + growth_rate)
                    trajectory.append(new_capability)
                
                country_trajectories[domain.value] = trajectory
            
            trajectories[country] = country_trajectories
        
        return trajectories
    
    def simulate_competitive_action(self, action: CompetitiveAction) -> Dict[str, Any]:
        """Simulate the impact of a competitive action."""
        
        impact_analysis = {
            "action": action,
            "direct_impacts": {},
            "competitive_responses": [],
            "market_effects": {},
            "strategic_implications": []
        }
        
        # Calculate direct impacts on actor capabilities
        actor_capabilities = self.national_capabilities.get(action.actor_country, {})
        if action.domain in actor_capabilities:
            current_cap = actor_capabilities[action.domain]
            
            # Investment impact on capability
            if action.action_type == CompetitiveStrategy.INNOVATION_LEADERSHIP:
                capability_boost = min(0.2, action.investment_billions / 50)
                impact_analysis["direct_impacts"]["capability_increase"] = capability_boost
                
            elif action.action_type == CompetitiveStrategy.INDIGENOUS_DEVELOPMENT:
                capability_boost = min(0.15, action.investment_billions / 75)
                impact_analysis["direct_impacts"]["capability_increase"] = capability_boost
                
            elif action.action_type == CompetitiveStrategy.EXPORT_RESTRICTIONS:
                # Negative impact on targets
                for target in action.target_countries:
                    impact_analysis["direct_impacts"][f"{target}_capability_reduction"] = -0.1
        
        # Predict competitive responses
        for target_country in action.target_countries:
            if target_country in self.national_capabilities:
                response_prob = action.countermeasure_probability
                if response_prob > 0.5:
                    impact_analysis["competitive_responses"].append({
                        "country": target_country,
                        "response_type": "Accelerated indigenous development",
                        "probability": response_prob,
                        "investment_increase": action.investment_billions * 0.5
                    })
        
        # Market effects
        if action.action_type == CompetitiveStrategy.EXPORT_RESTRICTIONS:
            impact_analysis["market_effects"] = {
                "market_fragmentation_increase": 0.2,
                "price_increase_targets": 0.15,
                "alternative_supplier_growth": 0.3
            }
        elif action.action_type == CompetitiveStrategy.COST_COMPETITION:
            impact_analysis["market_effects"] = {
                "price_pressure": 0.1,
                "market_share_gain": 0.05
            }
        
        # Strategic implications
        impact_analysis["strategic_implications"] = self._assess_strategic_implications(action)
        
        # Store action
        self.competitive_actions.append(action)
        
        return impact_analysis
    
    def _assess_strategic_implications(self, action: CompetitiveAction) -> List[str]:
        """Assess strategic implications of competitive actions."""
        implications = []
        
        if action.action_type == CompetitiveStrategy.EXPORT_RESTRICTIONS:
            implications.extend([
                "Potential acceleration of technology decoupling",
                "Increased incentives for indigenous development",
                "Risk of retaliatory measures"
            ])
            
            if "China" in action.target_countries and action.actor_country == "USA":
                implications.append("Escalation of US-China technology competition")
                
        elif action.action_type == CompetitiveStrategy.INDIGENOUS_DEVELOPMENT:
            implications.extend([
                "Reduced dependency on foreign technology",
                "Potential for technology standard divergence",
                "Increased national R&D capacity"
            ])
            
            if action.investment_billions > 50:
                implications.append("Major strategic technology initiative")
        
        elif action.action_type == CompetitiveStrategy.VERTICAL_INTEGRATION:
            implications.extend([
                "Improved supply chain control",
                "Potential cost advantages",
                "Reduced ecosystem collaboration"
            ])
        
        return implications
    
    def forecast_competitive_landscape(self, scenario: str, years: int = 10) -> Dict[str, Any]:
        """Forecast competitive landscape under different scenarios."""
        
        forecast = {
            "scenario": scenario,
            "time_horizon": years,
            "projected_leaders": {},
            "innovation_outcomes": {},
            "strategic_shifts": [],
            "risk_factors": []
        }
        
        if scenario == "continued_competition":
            # Current trends continue
            forecast["projected_leaders"] = self._project_current_trends(years)
            forecast["innovation_outcomes"] = self._project_innovation_races(years)
            
        elif scenario == "technology_decoupling":
            # Major powers develop separate technology ecosystems
            forecast["strategic_shifts"] = [
                "Fragmentation into US-led and China-led technology blocs",
                "Reduced global collaboration on technology standards",
                "Accelerated indigenous development programs"
            ]
            forecast["risk_factors"] = [
                "Slower overall pace of innovation",
                "Higher development costs due to duplication",
                "Increased geopolitical tensions"
            ]
            
        elif scenario == "breakthrough_disruption":
            # Major breakthrough changes competitive dynamics
            forecast["strategic_shifts"] = [
                "Rapid shift in competitive positions",
                "Obsolescence of current technology investments",
                "New strategic alliances formation"
            ]
        
        self.strategic_forecasts[scenario] = forecast
        return forecast
    
    def _project_current_trends(self, years: int) -> Dict[str, List[str]]:
        """Project current competitive trends forward."""
        projected_leaders = {}
        
        trajectories = self._project_capability_trajectories(years)
        
        for domain in CompetitionDomain:
            domain_projections = {}
            
            for country, country_trajectories in trajectories.items():
                if domain.value in country_trajectories:
                    final_capability = country_trajectories[domain.value][-1]
                    domain_projections[country] = final_capability
            
            # Sort by projected capability
            sorted_projections = sorted(domain_projections.items(), key=lambda x: x[1], reverse=True)
            projected_leaders[domain.value] = [country for country, _ in sorted_projections[:3]]
        
        return projected_leaders
    
    def _project_innovation_races(self, years: int) -> Dict[str, Dict[str, float]]:
        """Project outcomes of ongoing innovation races."""
        outcomes = {}
        
        for race_id, race in self.innovation_races.items():
            race_outcomes = {}
            
            for country, timeline in race.breakthrough_timeline.items():
                if timeline <= years:
                    # Breakthrough achieved within timeframe
                    success_prob = race.projected_outcomes.get(country, {}).get("takes_lead", 0)
                    success_prob += race.projected_outcomes.get(country, {}).get("breakthrough", 0)
                    race_outcomes[country] = success_prob
                else:
                    # Partial progress
                    progress_prob = race.projected_outcomes.get(country, {}).get("progress", 0.3)
                    race_outcomes[country] = progress_prob * 0.5
            
            outcomes[race_id] = race_outcomes
        
        return outcomes
    
    def get_competition_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategic competition summary."""
        
        # Ensure analysis is current
        if not self.competition_dynamics:
            self.analyze_competitive_dynamics()
        
        # Calculate key metrics
        total_global_investment = sum(
            sum(cap.r_and_d_investment_billions for cap in capabilities.values())
            for capabilities in self.national_capabilities.values()
        )
        
        # Competition intensity by domain
        domain_intensity = {}
        for domain in CompetitionDomain:
            competitors = []
            for country, capabilities in self.national_capabilities.items():
                if domain in capabilities:
                    competitors.append(capabilities[domain].current_capability)
            
            if competitors:
                intensity = 1 - np.var(competitors)  # Lower variance = higher intensity
                domain_intensity[domain.value] = intensity
        
        return {
            "competition_overview": {
                "total_global_rd_investment_billions": total_global_investment,
                "active_competitors": len(self.national_capabilities),
                "competition_domains": len(CompetitionDomain),
                "ongoing_innovation_races": len(self.innovation_races)
            },
            "competitive_landscape": self.competition_dynamics.get("current_leaders_by_domain", {}),
            "investment_analysis": self.competition_dynamics.get("investment_trends", {}),
            "domain_competition_intensity": domain_intensity,
            "innovation_race_status": {
                race_id: {
                    "leaders": race.current_leaders,
                    "total_investment": sum(race.investment_levels.values()),
                    "competition_intensity": len(race.participating_countries)
                }
                for race_id, race in self.innovation_races.items()
            },
            "recent_competitive_actions": len(self.competitive_actions),
            "strategic_forecasts": list(self.strategic_forecasts.keys())
        } 