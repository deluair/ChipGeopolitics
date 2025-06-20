"""
Economic Warfare Scenarios for Semiconductor Industry

Models economic warfare tactics including trade wars, sanctions, export controls,
supply chain weaponization, technology embargoes, and financial warfare in the
semiconductor sector.
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

class WarfareType(Enum):
    """Types of economic warfare tactics."""
    TRADE_TARIFFS = "trade_tariffs"
    EXPORT_CONTROLS = "export_controls"
    TECHNOLOGY_EMBARGO = "technology_embargo"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    FINANCIAL_SANCTIONS = "financial_sanctions"
    INVESTMENT_RESTRICTIONS = "investment_restrictions"
    STANDARDS_WARFARE = "standards_warfare"
    TALENT_RESTRICTIONS = "talent_restrictions"
    MARKET_ACCESS_DENIAL = "market_access_denial"
    SUBSIDY_WARFARE = "subsidy_warfare"

class EscalationLevel(Enum):
    """Escalation levels for economic warfare."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"
    EXTREME = "extreme"

@dataclass
class EconomicWarfareAction:
    """Economic warfare action definition."""
    action_id: str
    warfare_type: WarfareType
    initiator_country: str
    target_countries: List[str]
    escalation_level: EscalationLevel
    implementation_date: datetime
    duration_months: int
    target_sectors: List[str]  # Specific semiconductor sectors
    target_companies: List[str]  # Specific companies
    economic_impact_billions: float
    strategic_objectives: List[str]
    success_probability: float
    retaliation_probability: float
    collateral_damage: Dict[str, float]  # Country -> damage score

@dataclass
class WarfareScenario:
    """Economic warfare scenario analysis."""
    scenario_id: str
    scenario_name: str
    participating_countries: List[str]
    warfare_actions: List[EconomicWarfareAction]
    timeline_months: int
    escalation_path: List[EscalationLevel]
    economic_impacts: Dict[str, Dict[str, float]]  # Country -> sector -> impact
    geopolitical_consequences: List[str]
    technology_decoupling_level: float  # 0-1 decoupling degree
    supply_chain_fragmentation: float  # 0-1 fragmentation level

@dataclass
class CountryVulnerability:
    """Country vulnerability to economic warfare."""
    country: str
    economic_dependencies: Dict[str, float]  # Partner -> dependency score
    critical_imports: List[str]  # Critical semiconductor imports
    export_reliance: Dict[str, float]  # Market -> export reliance
    financial_vulnerabilities: List[str]
    supply_chain_chokepoints: List[str]
    technological_dependencies: Dict[str, float]  # Technology -> dependency
    resilience_score: float  # 0-1 overall resilience

class EconomicWarfareModel:
    """
    Models economic warfare scenarios in semiconductor industry.
    
    Analyzes:
    - Economic warfare tactics and effectiveness
    - Escalation dynamics and response patterns
    - Economic impact assessment
    - Supply chain weaponization
    - Technology decoupling scenarios
    - Retaliation and counter-measures
    - Long-term strategic consequences
    """
    
    def __init__(self):
        # Warfare state
        self.active_actions: List[EconomicWarfareAction] = []
        self.scenarios: Dict[str, WarfareScenario] = {}
        self.country_vulnerabilities: Dict[str, CountryVulnerability] = {}
        
        # Analysis results
        self.impact_assessments: Dict[str, Any] = {}
        self.escalation_models: Dict[str, Any] = {}
        self.resilience_analysis: Dict[str, Any] = {}
        
        # Initialize with current vulnerabilities and ongoing conflicts
        self._initialize_country_vulnerabilities()
        self._initialize_current_warfare_state()
    
    def _initialize_country_vulnerabilities(self):
        """Initialize country vulnerability profiles."""
        
        self.country_vulnerabilities = {
            "USA": CountryVulnerability(
                country="USA",
                economic_dependencies={
                    "Taiwan": 0.8,  # Advanced chip manufacturing
                    "China": 0.4,   # Assembly and testing
                    "South Korea": 0.6,  # Memory
                    "Japan": 0.5    # Materials
                },
                critical_imports=["advanced_chips", "rare_earth_materials", "assembly_services"],
                export_reliance={
                    "China": 0.3,
                    "Europe": 0.4,
                    "Asia_Pacific": 0.5
                },
                financial_vulnerabilities=["dollar_weaponization_backlash", "debt_ceiling_risks"],
                supply_chain_chokepoints=["Taiwan_Strait", "South_China_Sea", "Malacca_Strait"],
                technological_dependencies={
                    "EUV_lithography": 0.9,  # Netherlands dependency
                    "rare_earth_processing": 0.8,  # China dependency
                    "advanced_packaging": 0.6   # Asia dependency
                },
                resilience_score=0.7
            ),
            
            "China": CountryVulnerability(
                country="China",
                economic_dependencies={
                    "Taiwan": 0.9,      # Advanced chips
                    "USA": 0.7,         # Design tools, IP
                    "Netherlands": 0.95, # EUV lithography
                    "Japan": 0.8,       # Materials
                    "South Korea": 0.6   # Memory
                },
                critical_imports=["advanced_semiconductors", "design_software", "manufacturing_equipment"],
                export_reliance={
                    "USA": 0.4,
                    "Europe": 0.3,
                    "Global_South": 0.6
                },
                financial_vulnerabilities=["SWIFT_exclusion", "dollar_clearing", "foreign_investment_freeze"],
                supply_chain_chokepoints=["Strait_of_Hormuz", "Suez_Canal", "Pacific_shipping"],
                technological_dependencies={
                    "advanced_processes": 0.9,
                    "design_tools": 0.8,
                    "high_end_equipment": 0.85
                },
                resilience_score=0.4
            ),
            
            "Taiwan": CountryVulnerability(
                country="Taiwan",
                economic_dependencies={
                    "China": 0.6,   # Market access
                    "USA": 0.7,     # Technology, protection
                    "Japan": 0.6,   # Materials
                    "Netherlands": 0.8  # Equipment
                },
                critical_imports=["manufacturing_equipment", "materials", "design_tools"],
                export_reliance={
                    "China": 0.4,
                    "USA": 0.3,
                    "Global": 0.8  # High global dependence
                },
                financial_vulnerabilities=["limited_recognition", "china_pressure", "small_market"],
                supply_chain_chokepoints=["Taiwan_Strait", "Kaohsiung_Port", "air_transport"],
                technological_dependencies={
                    "equipment": 0.7,
                    "materials": 0.6,
                    "design_tools": 0.5
                },
                resilience_score=0.5
            ),
            
            "South Korea": CountryVulnerability(
                country="South Korea",
                economic_dependencies={
                    "China": 0.6,   # Market and materials
                    "USA": 0.6,     # Technology
                    "Japan": 0.5,   # Materials
                    "Taiwan": 0.4   # Foundry services
                },
                critical_imports=["advanced_logic_chips", "materials", "equipment"],
                export_reliance={
                    "China": 0.5,
                    "USA": 0.3,
                    "Global": 0.7
                },
                financial_vulnerabilities=["chaebol_concentration", "china_market_access"],
                supply_chain_chokepoints=["Korea_Strait", "China_land_border"],
                technological_dependencies={
                    "logic_chips": 0.4,
                    "advanced_equipment": 0.6
                },
                resilience_score=0.6
            ),
            
            "Japan": CountryVulnerability(
                country="Japan",
                economic_dependencies={
                    "China": 0.5,   # Market access
                    "USA": 0.6,     # Alliance, technology
                    "Taiwan": 0.4,  # Advanced chips
                    "South Korea": 0.3
                },
                critical_imports=["advanced_semiconductors", "rare_earth_materials"],
                export_reliance={
                    "China": 0.4,
                    "USA": 0.3,
                    "Asia": 0.6
                },
                financial_vulnerabilities=["aging_population", "debt_levels"],
                supply_chain_chokepoints=["Malacca_Strait", "South_China_Sea"],
                technological_dependencies={
                    "advanced_chips": 0.5,
                    "rare_earths": 0.7
                },
                resilience_score=0.7
            )
        }
    
    def _initialize_current_warfare_state(self):
        """Initialize current economic warfare actions."""
        
        # US-China trade war elements
        us_china_export_controls = EconomicWarfareAction(
            action_id="us_china_export_controls_2024",
            warfare_type=WarfareType.EXPORT_CONTROLS,
            initiator_country="USA",
            target_countries=["China"],
            escalation_level=EscalationLevel.HIGH,
            implementation_date=datetime(2022, 10, 1),
            duration_months=24,  # Ongoing
            target_sectors=["advanced_semiconductors", "AI_chips", "quantum_computing"],
            target_companies=["SMIC", "Huawei", "YMTC", "Loongson"],
            economic_impact_billions=50.0,
            strategic_objectives=["slow_china_ai_development", "maintain_tech_superiority", "security_concerns"],
            success_probability=0.7,
            retaliation_probability=0.8,
            collateral_damage={"Netherlands": 5.0, "South Korea": 3.0, "Japan": 2.0}
        )
        
        china_rare_earth_restrictions = EconomicWarfareAction(
            action_id="china_rare_earth_controls_2024",
            warfare_type=WarfareType.EXPORT_CONTROLS,
            initiator_country="China",
            target_countries=["USA", "Japan", "Netherlands"],
            escalation_level=EscalationLevel.MODERATE,
            implementation_date=datetime(2023, 7, 1),
            duration_months=12,
            target_sectors=["materials", "manufacturing_equipment"],
            target_companies=["applied_materials", "ASML", "Tokyo_Electron"],
            economic_impact_billions=15.0,
            strategic_objectives=["counter_us_controls", "leverage_material_dominance"],
            success_probability=0.6,
            retaliation_probability=0.5,
            collateral_damage={"Taiwan": 2.0, "South Korea": 1.5}
        )
        
        self.active_actions = [us_china_export_controls, china_rare_earth_restrictions]
    
    def analyze_warfare_effectiveness(self, action: EconomicWarfareAction) -> Dict[str, Any]:
        """Analyze effectiveness of an economic warfare action."""
        
        effectiveness_analysis = {
            "action": action,
            "direct_impact": {},
            "indirect_effects": {},
            "adaptation_responses": [],
            "long_term_consequences": [],
            "effectiveness_score": 0.0
        }
        
        # Calculate direct economic impact
        for target_country in action.target_countries:
            if target_country in self.country_vulnerabilities:
                vulnerability = self.country_vulnerabilities[target_country]
                
                # Impact based on warfare type and country vulnerability
                if action.warfare_type == WarfareType.EXPORT_CONTROLS:
                    tech_dependency = vulnerability.technological_dependencies.get("advanced_processes", 0.5)
                    impact_multiplier = tech_dependency * (action.escalation_level.value == "high" and 1.5 or 1.0)
                    direct_impact = action.economic_impact_billions * impact_multiplier
                    
                elif action.warfare_type == WarfareType.SUPPLY_CHAIN_DISRUPTION:
                    supply_vulnerability = len(vulnerability.supply_chain_chokepoints) / 5.0  # Normalize
                    direct_impact = action.economic_impact_billions * supply_vulnerability
                    
                elif action.warfare_type == WarfareType.FINANCIAL_SANCTIONS:
                    financial_vulnerability = len(vulnerability.financial_vulnerabilities) / 3.0
                    direct_impact = action.economic_impact_billions * financial_vulnerability
                    
                else:
                    direct_impact = action.economic_impact_billions * 0.5  # Default
                
                effectiveness_analysis["direct_impact"][target_country] = direct_impact
        
        # Analyze indirect effects and adaptations
        for target_country in action.target_countries:
            adaptation_responses = self._predict_adaptation_responses(action, target_country)
            effectiveness_analysis["adaptation_responses"].extend(adaptation_responses)
        
        # Calculate overall effectiveness
        total_direct_impact = sum(effectiveness_analysis["direct_impact"].values())
        adaptation_resistance = len(effectiveness_analysis["adaptation_responses"]) * 0.1
        
        effectiveness_analysis["effectiveness_score"] = min(1.0, 
            (total_direct_impact / 100.0) * action.success_probability - adaptation_resistance)
        
        # Long-term consequences
        if action.escalation_level in [EscalationLevel.HIGH, EscalationLevel.SEVERE]:
            effectiveness_analysis["long_term_consequences"].extend([
                "Technology decoupling acceleration",
                "Supply chain fragmentation",
                "Innovation ecosystem bifurcation"
            ])
        
        if action.retaliation_probability > 0.6:
            effectiveness_analysis["long_term_consequences"].append("Escalation spiral risk")
        
        return effectiveness_analysis
    
    def _predict_adaptation_responses(self, action: EconomicWarfareAction, target_country: str) -> List[str]:
        """Predict how target country adapts to economic warfare."""
        
        responses = []
        
        if action.warfare_type == WarfareType.EXPORT_CONTROLS:
            responses.extend([
                "Indigenous technology development acceleration",
                "Alternative supplier partnerships",
                "Technology smuggling and circumvention",
                "Stockpiling critical components"
            ])
        
        elif action.warfare_type == WarfareType.SUPPLY_CHAIN_DISRUPTION:
            responses.extend([
                "Supply chain diversification",
                "Strategic reserve building",
                "Alternative route development",
                "Regional partnership strengthening"
            ])
        
        elif action.warfare_type == WarfareType.FINANCIAL_SANCTIONS:
            responses.extend([
                "Alternative payment system development",
                "Currency diversification",
                "Financial institution restructuring",
                "Bilateral clearing agreements"
            ])
        
        # Country-specific adaptations
        if target_country == "China":
            responses.extend([
                "State-led technology investment surge",
                "Belt and Road technology integration",
                "Domestic market consolidation"
            ])
        elif target_country == "USA":
            responses.extend([
                "Allied coordination strengthening",
                "Domestic manufacturing reshoring",
                "STEM education investment"
            ])
        
        return responses
    
    def simulate_escalation_scenario(self, scenario_name: str, initial_action: EconomicWarfareAction) -> WarfareScenario:
        """Simulate escalation scenario from initial action."""
        
        scenario = WarfareScenario(
            scenario_id=f"escalation_{scenario_name}",
            scenario_name=scenario_name,
            participating_countries=[initial_action.initiator_country] + initial_action.target_countries,
            warfare_actions=[initial_action],
            timeline_months=24,
            escalation_path=[initial_action.escalation_level],
            economic_impacts={},
            geopolitical_consequences=[],
            technology_decoupling_level=0.0,
            supply_chain_fragmentation=0.0
        )
        
        current_month = 0
        current_escalation = initial_action.escalation_level
        
        # Simulate escalation dynamics
        while current_month < scenario.timeline_months:
            # Check for retaliation probability
            for target_country in initial_action.target_countries:
                if np.random.random() < initial_action.retaliation_probability / 12:  # Monthly probability
                    
                    # Generate retaliation action
                    retaliation = self._generate_retaliation_action(
                        target_country, initial_action.initiator_country, current_escalation
                    )
                    scenario.warfare_actions.append(retaliation)
                    
                    # Escalation level increase
                    current_escalation = self._escalate_level(current_escalation)
                    scenario.escalation_path.append(current_escalation)
            
            current_month += 3  # Quarterly simulation
        
        # Calculate cumulative impacts
        scenario.economic_impacts = self._calculate_scenario_impacts(scenario)
        scenario.geopolitical_consequences = self._assess_geopolitical_consequences(scenario)
        scenario.technology_decoupling_level = self._calculate_decoupling_level(scenario)
        scenario.supply_chain_fragmentation = self._calculate_fragmentation_level(scenario)
        
        self.scenarios[scenario.scenario_id] = scenario
        return scenario
    
    def _generate_retaliation_action(self, retaliator: str, target: str, 
                                   current_escalation: EscalationLevel) -> EconomicWarfareAction:
        """Generate retaliation action."""
        
        # Select retaliation type based on retaliator capabilities
        retaliator_profile = self.country_vulnerabilities.get(retaliator)
        
        if retaliator == "China":
            warfare_type = WarfareType.EXPORT_CONTROLS  # Rare earth controls
            target_sectors = ["materials", "rare_earth_elements"]
        elif retaliator == "USA":
            warfare_type = WarfareType.FINANCIAL_SANCTIONS
            target_sectors = ["financial_services", "technology_transfer"]
        else:
            warfare_type = WarfareType.TRADE_TARIFFS
            target_sectors = ["semiconductors", "electronics"]
        
        return EconomicWarfareAction(
            action_id=f"retaliation_{retaliator}_{target}_{datetime.now().timestamp()}",
            warfare_type=warfare_type,
            initiator_country=retaliator,
            target_countries=[target],
            escalation_level=current_escalation,
            implementation_date=datetime.now(),
            duration_months=12,
            target_sectors=target_sectors,
            target_companies=[],
            economic_impact_billions=20.0,
            strategic_objectives=["counter_economic_warfare", "demonstrate_resolve"],
            success_probability=0.6,
            retaliation_probability=0.7,
            collateral_damage={}
        )
    
    def _escalate_level(self, current_level: EscalationLevel) -> EscalationLevel:
        """Escalate warfare level."""
        escalation_mapping = {
            EscalationLevel.LOW: EscalationLevel.MODERATE,
            EscalationLevel.MODERATE: EscalationLevel.HIGH,
            EscalationLevel.HIGH: EscalationLevel.SEVERE,
            EscalationLevel.SEVERE: EscalationLevel.EXTREME,
            EscalationLevel.EXTREME: EscalationLevel.EXTREME  # Maximum
        }
        return escalation_mapping.get(current_level, current_level)
    
    def _calculate_scenario_impacts(self, scenario: WarfareScenario) -> Dict[str, Dict[str, float]]:
        """Calculate cumulative economic impacts of scenario."""
        
        impacts = {}
        
        for country in scenario.participating_countries:
            impacts[country] = {
                "gdp_impact": 0.0,
                "trade_disruption": 0.0,
                "technology_access": 0.0,
                "supply_chain_costs": 0.0
            }
        
        # Aggregate impacts from all actions
        for action in scenario.warfare_actions:
            for target_country in action.target_countries:
                if target_country in impacts:
                    # Scale impact by escalation level
                    escalation_multiplier = {
                        EscalationLevel.LOW: 0.5,
                        EscalationLevel.MODERATE: 1.0,
                        EscalationLevel.HIGH: 1.5,
                        EscalationLevel.SEVERE: 2.0,
                        EscalationLevel.EXTREME: 3.0
                    }.get(action.escalation_level, 1.0)
                    
                    impact_value = action.economic_impact_billions * escalation_multiplier / 1000  # As % of GDP
                    
                    if action.warfare_type == WarfareType.EXPORT_CONTROLS:
                        impacts[target_country]["technology_access"] += impact_value
                    elif action.warfare_type == WarfareType.SUPPLY_CHAIN_DISRUPTION:
                        impacts[target_country]["supply_chain_costs"] += impact_value
                    elif action.warfare_type == WarfareType.TRADE_TARIFFS:
                        impacts[target_country]["trade_disruption"] += impact_value
                    
                    impacts[target_country]["gdp_impact"] += impact_value
        
        return impacts
    
    def _assess_geopolitical_consequences(self, scenario: WarfareScenario) -> List[str]:
        """Assess geopolitical consequences of warfare scenario."""
        
        consequences = []
        
        max_escalation = max(scenario.escalation_path)
        
        if max_escalation in [EscalationLevel.HIGH, EscalationLevel.SEVERE]:
            consequences.extend([
                "Bilateral relationship deterioration",
                "Alliance system strain",
                "Third-party forced alignment",
                "International institution erosion"
            ])
        
        if max_escalation == EscalationLevel.EXTREME:
            consequences.extend([
                "Complete economic decoupling",
                "Technology cold war",
                "Proxy economic conflicts",
                "Global system fragmentation"
            ])
        
        # Check for US-China specifically
        if "USA" in scenario.participating_countries and "China" in scenario.participating_countries:
            consequences.extend([
                "US-China strategic competition intensification",
                "Taiwan situation deterioration risk",
                "ASEAN neutrality pressure",
                "Global supply chain bifurcation"
            ])
        
        return consequences
    
    def _calculate_decoupling_level(self, scenario: WarfareScenario) -> float:
        """Calculate technology decoupling level from scenario."""
        
        decoupling_score = 0.0
        
        for action in scenario.warfare_actions:
            if action.warfare_type in [WarfareType.EXPORT_CONTROLS, WarfareType.TECHNOLOGY_EMBARGO]:
                escalation_weight = {
                    EscalationLevel.LOW: 0.1,
                    EscalationLevel.MODERATE: 0.2,
                    EscalationLevel.HIGH: 0.4,
                    EscalationLevel.SEVERE: 0.6,
                    EscalationLevel.EXTREME: 0.8
                }.get(action.escalation_level, 0.2)
                
                decoupling_score += escalation_weight
        
        return min(1.0, decoupling_score)
    
    def _calculate_fragmentation_level(self, scenario: WarfareScenario) -> float:
        """Calculate supply chain fragmentation level."""
        
        fragmentation_score = 0.0
        
        for action in scenario.warfare_actions:
            if action.warfare_type in [WarfareType.SUPPLY_CHAIN_DISRUPTION, WarfareType.EXPORT_CONTROLS]:
                escalation_weight = {
                    EscalationLevel.LOW: 0.1,
                    EscalationLevel.MODERATE: 0.2,
                    EscalationLevel.HIGH: 0.3,
                    EscalationLevel.SEVERE: 0.5,
                    EscalationLevel.EXTREME: 0.7
                }.get(action.escalation_level, 0.2)
                
                fragmentation_score += escalation_weight
        
        return min(1.0, fragmentation_score)
    
    def assess_country_resilience(self, country: str, warfare_types: List[WarfareType]) -> Dict[str, Any]:
        """Assess country resilience to specific warfare types."""
        
        if country not in self.country_vulnerabilities:
            return {"error": "Country not found"}
        
        vulnerability = self.country_vulnerabilities[country]
        
        resilience_assessment = {
            "country": country,
            "overall_resilience": vulnerability.resilience_score,
            "specific_vulnerabilities": {},
            "mitigation_strategies": [],
            "critical_dependencies": [],
            "resilience_gaps": []
        }
        
        for warfare_type in warfare_types:
            if warfare_type == WarfareType.EXPORT_CONTROLS:
                tech_vulnerability = max(vulnerability.technological_dependencies.values())
                resilience_assessment["specific_vulnerabilities"]["export_controls"] = tech_vulnerability
                
            elif warfare_type == WarfareType.SUPPLY_CHAIN_DISRUPTION:
                supply_vulnerability = len(vulnerability.supply_chain_chokepoints) / 5.0
                resilience_assessment["specific_vulnerabilities"]["supply_chain"] = supply_vulnerability
                
            elif warfare_type == WarfareType.FINANCIAL_SANCTIONS:
                financial_vulnerability = len(vulnerability.financial_vulnerabilities) / 3.0
                resilience_assessment["specific_vulnerabilities"]["financial"] = financial_vulnerability
        
        # Identify critical dependencies
        high_dependencies = [k for k, v in vulnerability.economic_dependencies.items() if v > 0.7]
        resilience_assessment["critical_dependencies"] = high_dependencies
        
        # Suggest mitigation strategies
        if vulnerability.resilience_score < 0.6:
            resilience_assessment["mitigation_strategies"].extend([
                "Diversify supply chain sources",
                "Build strategic reserves",
                "Develop alternative technologies",
                "Strengthen alliance partnerships"
            ])
        
        # Identify gaps
        if tech_vulnerability > 0.7:
            resilience_assessment["resilience_gaps"].append("High technology dependency")
        if len(high_dependencies) > 2:
            resilience_assessment["resilience_gaps"].append("Excessive partner concentration")
        
        return resilience_assessment
    
    def get_warfare_summary(self) -> Dict[str, Any]:
        """Get comprehensive economic warfare summary."""
        
        # Current warfare state
        active_warfare_types = list(set(action.warfare_type.value for action in self.active_actions))
        affected_countries = list(set([action.initiator_country] + 
                                   [country for action in self.active_actions 
                                    for country in action.target_countries]))
        
        # Vulnerability analysis
        most_vulnerable = min(self.country_vulnerabilities.items(), 
                            key=lambda x: x[1].resilience_score, default=(None, None))
        least_vulnerable = max(self.country_vulnerabilities.items(), 
                             key=lambda x: x[1].resilience_score, default=(None, None))
        
        # Economic impact
        total_economic_impact = sum(action.economic_impact_billions for action in self.active_actions)
        
        return {
            "current_warfare_state": {
                "active_actions": len(self.active_actions),
                "warfare_types": active_warfare_types,
                "affected_countries": affected_countries,
                "total_economic_impact_billions": total_economic_impact,
                "highest_escalation": max((action.escalation_level for action in self.active_actions), 
                                        default=EscalationLevel.LOW).value
            },
            "vulnerability_landscape": {
                "countries_analyzed": len(self.country_vulnerabilities),
                "most_vulnerable": most_vulnerable[0] if most_vulnerable[0] else None,
                "least_vulnerable": least_vulnerable[0] if least_vulnerable[0] else None,
                "average_resilience": np.mean([v.resilience_score for v in self.country_vulnerabilities.values()])
            },
            "scenario_analysis": {
                "scenarios_simulated": len(self.scenarios),
                "max_decoupling_level": max((s.technology_decoupling_level for s in self.scenarios.values()), 
                                          default=0.0),
                "max_fragmentation_level": max((s.supply_chain_fragmentation for s in self.scenarios.values()), 
                                             default=0.0)
            },
            "critical_dependencies": {
                country: [dep for dep, score in vulnerability.economic_dependencies.items() if score > 0.7]
                for country, vulnerability in self.country_vulnerabilities.items()
            },
            "warfare_effectiveness": {
                action.action_id: self.analyze_warfare_effectiveness(action)["effectiveness_score"]
                for action in self.active_actions
            }
        } 