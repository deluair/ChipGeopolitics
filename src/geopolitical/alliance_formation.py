"""
Alliance Formation Dynamics for Semiconductor Geopolitics

Models the formation, evolution, and dissolution of strategic alliances in the semiconductor
industry including technology partnerships, supply chain alliances, standards coalitions,
and defense/security partnerships.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class AllianceType(Enum):
    """Types of strategic alliances."""
    TECHNOLOGY_PARTNERSHIP = "technology_partnership"
    SUPPLY_CHAIN_ALLIANCE = "supply_chain_alliance"
    STANDARDS_COALITION = "standards_coalition"
    DEFENSE_SECURITY = "defense_security"
    TRADE_BLOC = "trade_bloc"
    RESEARCH_CONSORTIUM = "research_consortium"
    EXPORT_CONTROL_COOPERATION = "export_control_cooperation"
    MARKET_ACCESS_AGREEMENT = "market_access_agreement"

class AllianceStrength(Enum):
    """Alliance relationship strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    STRATEGIC = "strategic"

@dataclass
class Alliance:
    """Strategic alliance between countries/regions."""
    alliance_id: str
    alliance_type: AllianceType
    members: List[str]  # Country codes
    leader_country: Optional[str]  # Dominant member if any
    strength: AllianceStrength
    formation_date: datetime
    objectives: List[str]
    shared_interests: List[str]
    conflicting_interests: List[str]
    technology_domains: List[str]  # Focus areas
    economic_benefits: Dict[str, float]  # Member -> benefit score
    strategic_value: float  # 0-1 alliance strategic value
    stability_score: float  # 0-1 stability indicator
    exclusivity_level: float  # 0-1 how exclusive vs open

@dataclass
class AllianceProposal:
    """Proposed new alliance."""
    proposal_id: str
    proposer_country: str
    target_countries: List[str]
    alliance_type: AllianceType
    objectives: List[str]
    proposed_benefits: Dict[str, float]
    formation_probability: float
    timeline_months: int
    strategic_rationale: str

@dataclass
class CountryProfile:
    """Country profile for alliance analysis."""
    country: str
    economic_power: float  # 0-1 economic influence
    technological_capability: float  # 0-1 tech capability
    geopolitical_alignment: Dict[str, float]  # Country -> alignment score
    alliance_preference: Dict[AllianceType, float]  # Type -> preference
    current_commitments: List[str]  # Existing alliance IDs
    strategic_priorities: List[str]
    trust_scores: Dict[str, float]  # Country -> trust level
    domestic_constraints: List[str]

class AllianceFormationModel:
    """
    Models alliance formation dynamics in semiconductor geopolitics.
    
    Analyzes:
    - Alliance formation incentives and barriers
    - Member compatibility and trust
    - Strategic value calculations
    - Stability and evolution dynamics
    - Network effects and cascading alliances
    - Competitive alliance responses
    """
    
    def __init__(self):
        # Alliance network state
        self.alliances: Dict[str, Alliance] = {}
        self.country_profiles: Dict[str, CountryProfile] = {}
        self.alliance_proposals: List[AllianceProposal] = []
        
        # Analysis results
        self.alliance_network: nx.Graph = nx.Graph()
        self.stability_analysis: Dict[str, Any] = {}
        self.formation_predictions: Dict[str, float] = {}
        
        # Initialize with current 2024 alliance landscape
        self._initialize_country_profiles()
        self._initialize_existing_alliances()
    
    def _initialize_country_profiles(self):
        """Initialize country profiles for alliance analysis."""
        
        self.country_profiles = {
            "USA": CountryProfile(
                country="USA",
                economic_power=0.9,
                technological_capability=0.85,
                geopolitical_alignment={
                    "Taiwan": 0.8, "South Korea": 0.75, "Japan": 0.8, "Netherlands": 0.7,
                    "China": 0.1, "Russia": 0.0
                },
                alliance_preference={
                    AllianceType.TECHNOLOGY_PARTNERSHIP: 0.8,
                    AllianceType.DEFENSE_SECURITY: 0.9,
                    AllianceType.EXPORT_CONTROL_COOPERATION: 0.85,
                    AllianceType.STANDARDS_COALITION: 0.7
                },
                current_commitments=["quad_tech", "chips_act_allies", "export_control_regime"],
                strategic_priorities=["maintain_tech_leadership", "counter_china", "secure_supply_chains"],
                trust_scores={
                    "Taiwan": 0.8, "South Korea": 0.75, "Japan": 0.85, "Netherlands": 0.7,
                    "China": 0.2, "Russia": 0.1
                },
                domestic_constraints=["congressional_approval", "industry_resistance"]
            ),
            
            "China": CountryProfile(
                country="China",
                economic_power=0.8,
                technological_capability=0.6,
                geopolitical_alignment={
                    "Russia": 0.6, "North Korea": 0.7, "Iran": 0.5,
                    "USA": 0.1, "Taiwan": 0.0, "Japan": 0.3
                },
                alliance_preference={
                    AllianceType.TECHNOLOGY_PARTNERSHIP: 0.9,
                    AllianceType.SUPPLY_CHAIN_ALLIANCE: 0.8,
                    AllianceType.TRADE_BLOC: 0.7,
                    AllianceType.RESEARCH_CONSORTIUM: 0.85
                },
                current_commitments=["bri_tech", "shanghai_cooperation", "brics_plus"],
                strategic_priorities=["technology_independence", "counter_us_containment", "build_alternative_system"],
                trust_scores={
                    "Russia": 0.6, "North Korea": 0.5, "Iran": 0.4,
                    "USA": 0.1, "Taiwan": 0.0, "Japan": 0.2
                },
                domestic_constraints=["party_approval", "nationalist_sentiment"]
            ),
            
            "Taiwan": CountryProfile(
                country="Taiwan",
                economic_power=0.3,
                technological_capability=0.9,
                geopolitical_alignment={
                    "USA": 0.8, "Japan": 0.7, "South Korea": 0.6,
                    "China": 0.0
                },
                alliance_preference={
                    AllianceType.TECHNOLOGY_PARTNERSHIP: 0.9,
                    AllianceType.DEFENSE_SECURITY: 0.8,
                    AllianceType.SUPPLY_CHAIN_ALLIANCE: 0.85
                },
                current_commitments=["taiwan_us_tech", "quad_tech_cooperation"],
                strategic_priorities=["maintain_independence", "secure_partnerships", "technology_leadership"],
                trust_scores={
                    "USA": 0.8, "Japan": 0.75, "South Korea": 0.6,
                    "China": 0.0
                },
                domestic_constraints=["political_consensus", "business_interests"]
            ),
            
            "South Korea": CountryProfile(
                country="South Korea",
                economic_power=0.4,
                technological_capability=0.75,
                geopolitical_alignment={
                    "USA": 0.75, "Japan": 0.5, "Taiwan": 0.6,
                    "China": 0.4, "North Korea": 0.0
                },
                alliance_preference={
                    AllianceType.TECHNOLOGY_PARTNERSHIP: 0.8,
                    AllianceType.DEFENSE_SECURITY: 0.7,
                    AllianceType.SUPPLY_CHAIN_ALLIANCE: 0.75
                },
                current_commitments=["us_korea_alliance", "chips_act_cooperation"],
                strategic_priorities=["balance_us_china", "protect_tech_companies", "regional_stability"],
                trust_scores={
                    "USA": 0.75, "Japan": 0.5, "Taiwan": 0.6,
                    "China": 0.3, "North Korea": 0.0
                },
                domestic_constraints=["public_opinion", "china_economic_ties"]
            ),
            
            "Japan": CountryProfile(
                country="Japan",
                economic_power=0.5,
                technological_capability=0.7,
                geopolitical_alignment={
                    "USA": 0.8, "Taiwan": 0.7, "South Korea": 0.5,
                    "China": 0.3, "Russia": 0.1
                },
                alliance_preference={
                    AllianceType.TECHNOLOGY_PARTNERSHIP: 0.8,
                    AllianceType.DEFENSE_SECURITY: 0.75,
                    AllianceType.STANDARDS_COALITION: 0.7
                },
                current_commitments=["us_japan_alliance", "quad_cooperation"],
                strategic_priorities=["strengthen_us_alliance", "counter_china", "secure_supply_chains"],
                trust_scores={
                    "USA": 0.8, "Taiwan": 0.7, "South Korea": 0.5,
                    "China": 0.25, "Russia": 0.1
                },
                domestic_constraints=["pacifist_constitution", "business_lobbies"]
            )
        }
    
    def _initialize_existing_alliances(self):
        """Initialize current alliance landscape."""
        
        # US-led technology alliance
        self.alliances["quad_tech"] = Alliance(
            alliance_id="quad_tech",
            alliance_type=AllianceType.TECHNOLOGY_PARTNERSHIP,
            members=["USA", "Japan", "Taiwan", "South Korea"],
            leader_country="USA",
            strength=AllianceStrength.STRONG,
            formation_date=datetime(2022, 1, 1),
            objectives=["coordinate_semiconductor_policy", "secure_supply_chains", "maintain_tech_leadership"],
            shared_interests=["counter_china_tech", "supply_security", "innovation_leadership"],
            conflicting_interests=["market_access", "subsidy_competition", "technology_transfer"],
            technology_domains=["advanced_semiconductors", "AI", "quantum", "5G"],
            economic_benefits={
                "USA": 0.8, "Japan": 0.6, "Taiwan": 0.9, "South Korea": 0.7
            },
            strategic_value=0.85,
            stability_score=0.75,
            exclusivity_level=0.8
        )
        
        # CHIPS Act alliance
        self.alliances["chips_act_allies"] = Alliance(
            alliance_id="chips_act_allies",
            alliance_type=AllianceType.SUPPLY_CHAIN_ALLIANCE,
            members=["USA", "Taiwan", "South Korea"],
            leader_country="USA",
            strength=AllianceStrength.STRATEGIC,
            formation_date=datetime(2022, 8, 1),
            objectives=["secure_semiconductor_supply", "reduce_china_dependence", "build_resilient_ecosystem"],
            shared_interests=["supply_security", "china_containment", "technology_control"],
            conflicting_interests=["subsidy_competition", "market_share", "technology_access"],
            technology_domains=["advanced_manufacturing", "packaging", "materials"],
            economic_benefits={
                "USA": 0.7, "Taiwan": 0.8, "South Korea": 0.6
            },
            strategic_value=0.9,
            stability_score=0.8,
            exclusivity_level=0.9
        )
        
        # Export control regime
        self.alliances["export_control_regime"] = Alliance(
            alliance_id="export_control_regime",
            alliance_type=AllianceType.EXPORT_CONTROL_COOPERATION,
            members=["USA", "Netherlands", "Japan"],
            leader_country="USA",
            strength=AllianceStrength.STRONG,
            formation_date=datetime(2022, 10, 1),
            objectives=["restrict_china_access", "coordinate_export_controls", "maintain_technological_edge"],
            shared_interests=["china_containment", "technology_security", "competitive_advantage"],
            conflicting_interests=["economic_interests", "business_pressure", "implementation_costs"],
            technology_domains=["lithography", "manufacturing_equipment", "design_software"],
            economic_benefits={
                "USA": 0.6, "Netherlands": 0.4, "Japan": 0.5
            },
            strategic_value=0.8,
            stability_score=0.6,
            exclusivity_level=0.7
        )
        
        # Build alliance network
        self._build_alliance_network()
    
    def _build_alliance_network(self):
        """Build network graph of alliance relationships."""
        
        self.alliance_network.clear()
        
        # Add countries as nodes
        for country in self.country_profiles.keys():
            self.alliance_network.add_node(country)
        
        # Add alliance relationships as edges
        for alliance in self.alliances.values():
            members = alliance.members
            # Create edges between all alliance members
            for i, member1 in enumerate(members):
                for member2 in members[i+1:]:
                    weight = alliance.strength.value
                    if self.alliance_network.has_edge(member1, member2):
                        # Strengthen existing relationship
                        current_weight = self.alliance_network[member1][member2]['weight']
                        new_weight = min(1.0, current_weight + 0.3)
                        self.alliance_network[member1][member2]['weight'] = new_weight
                    else:
                        self.alliance_network.add_edge(member1, member2, 
                                                    weight=0.5, 
                                                    alliances=[alliance.alliance_id])
    
    def analyze_formation_incentives(self, country1: str, country2: str, 
                                   alliance_type: AllianceType) -> Dict[str, Any]:
        """Analyze incentives for alliance formation between two countries."""
        
        if country1 not in self.country_profiles or country2 not in self.country_profiles:
            return {"error": "Country not found in profiles"}
        
        profile1 = self.country_profiles[country1]
        profile2 = self.country_profiles[country2]
        
        analysis = {
            "compatibility_score": 0.0,
            "mutual_benefits": {},
            "shared_interests": [],
            "potential_conflicts": [],
            "formation_probability": 0.0,
            "strategic_rationale": "",
            "obstacles": []
        }
        
        # Calculate compatibility based on geopolitical alignment
        alignment1_to_2 = profile1.geopolitical_alignment.get(country2, 0.5)
        alignment2_to_1 = profile2.geopolitical_alignment.get(country1, 0.5)
        geopolitical_compatibility = (alignment1_to_2 + alignment2_to_1) / 2
        
        # Trust factor
        trust1_to_2 = profile1.trust_scores.get(country2, 0.5)
        trust2_to_1 = profile2.trust_scores.get(country1, 0.5)
        trust_factor = (trust1_to_2 + trust2_to_1) / 2
        
        # Alliance type preference
        type_pref1 = profile1.alliance_preference.get(alliance_type, 0.5)
        type_pref2 = profile2.alliance_preference.get(alliance_type, 0.5)
        type_compatibility = (type_pref1 + type_pref2) / 2
        
        # Calculate overall compatibility
        analysis["compatibility_score"] = (geopolitical_compatibility * 0.4 + 
                                         trust_factor * 0.3 + 
                                         type_compatibility * 0.3)
        
        # Identify mutual benefits
        if alliance_type == AllianceType.TECHNOLOGY_PARTNERSHIP:
            tech_gap = abs(profile1.technological_capability - profile2.technological_capability)
            if tech_gap > 0.1:  # Complementary capabilities
                analysis["mutual_benefits"]["technology_complementarity"] = min(0.3, tech_gap)
        
        if alliance_type == AllianceType.SUPPLY_CHAIN_ALLIANCE:
            # Economic complementarity
            econ_benefit = min(profile1.economic_power, profile2.economic_power) * 0.5
            analysis["mutual_benefits"]["supply_chain_security"] = econ_benefit
        
        # Shared strategic interests
        common_priorities = set(profile1.strategic_priorities) & set(profile2.strategic_priorities)
        analysis["shared_interests"] = list(common_priorities)
        
        # Potential conflicts from domestic constraints
        analysis["potential_conflicts"] = list(set(profile1.domestic_constraints + profile2.domestic_constraints))
        
        # Calculate formation probability
        base_probability = analysis["compatibility_score"]
        
        # Boost for shared interests
        if common_priorities:
            base_probability += len(common_priorities) * 0.1
        
        # Reduce for existing conflicting commitments
        existing_conflicts = self._check_commitment_conflicts(country1, country2)
        base_probability -= len(existing_conflicts) * 0.15
        
        analysis["formation_probability"] = max(0.0, min(1.0, base_probability))
        
        # Strategic rationale
        if analysis["formation_probability"] > 0.6:
            analysis["strategic_rationale"] = f"Strong alignment and complementary capabilities in {alliance_type.value}"
        elif analysis["formation_probability"] > 0.4:
            analysis["strategic_rationale"] = f"Moderate potential for {alliance_type.value} cooperation"
        else:
            analysis["strategic_rationale"] = f"Limited incentives for {alliance_type.value} alliance"
        
        # Identify obstacles
        if geopolitical_compatibility < 0.4:
            analysis["obstacles"].append("Geopolitical tensions")
        if trust_factor < 0.3:
            analysis["obstacles"].append("Trust deficit")
        if existing_conflicts:
            analysis["obstacles"].append("Conflicting existing commitments")
        
        return analysis
    
    def _check_commitment_conflicts(self, country1: str, country2: str) -> List[str]:
        """Check for conflicting existing commitments."""
        conflicts = []
        
        profile1 = self.country_profiles[country1]
        profile2 = self.country_profiles[country2]
        
        # Check if countries are in competing alliances
        for alliance_id, alliance in self.alliances.items():
            if country1 in alliance.members and country2 not in alliance.members:
                if alliance.exclusivity_level > 0.7:
                    conflicts.append(f"Exclusive alliance: {alliance_id}")
        
        return conflicts
    
    def predict_alliance_formation(self, time_horizon_months: int = 24) -> List[AllianceProposal]:
        """Predict potential new alliance formations."""
        
        predictions = []
        
        # Analyze all country pairs for alliance potential
        countries = list(self.country_profiles.keys())
        
        for i, country1 in enumerate(countries):
            for country2 in countries[i+1:]:
                # Skip if already strongly allied
                if self.alliance_network.has_edge(country1, country2):
                    if self.alliance_network[country1][country2]['weight'] > 0.7:
                        continue
                
                # Test different alliance types
                for alliance_type in AllianceType:
                    analysis = self.analyze_formation_incentives(country1, country2, alliance_type)
                    
                    if analysis.get("formation_probability", 0) > 0.5:
                        proposal = AllianceProposal(
                            proposal_id=f"{country1}_{country2}_{alliance_type.value}",
                            proposer_country=country1,  # Assume stronger partner proposes
                            target_countries=[country2],
                            alliance_type=alliance_type,
                            objectives=analysis.get("shared_interests", []),
                            proposed_benefits=analysis.get("mutual_benefits", {}),
                            formation_probability=analysis["formation_probability"],
                            timeline_months=int(24 / analysis["formation_probability"]),
                            strategic_rationale=analysis["strategic_rationale"]
                        )
                        predictions.append(proposal)
        
        # Sort by probability and return top candidates
        predictions.sort(key=lambda x: x.formation_probability, reverse=True)
        self.alliance_proposals = predictions[:10]  # Top 10 most likely
        
        return self.alliance_proposals
    
    def simulate_alliance_evolution(self, alliance_id: str, scenario: str, years: int = 5) -> Dict[str, Any]:
        """Simulate how an alliance evolves under different scenarios."""
        
        if alliance_id not in self.alliances:
            return {"error": "Alliance not found"}
        
        alliance = self.alliances[alliance_id]
        evolution = {
            "alliance_id": alliance_id,
            "scenario": scenario,
            "current_state": alliance,
            "projected_changes": {},
            "stability_forecast": [],
            "membership_changes": [],
            "strategic_impact": {}
        }
        
        if scenario == "geopolitical_tension":
            # Increased tension reduces alliance stability
            stability_decline = 0.1 * years
            new_stability = max(0.0, alliance.stability_score - stability_decline)
            
            evolution["projected_changes"]["stability_score"] = new_stability
            evolution["projected_changes"]["strength"] = self._adjust_alliance_strength(alliance.strength, -0.2)
            
            # Risk of member defection
            for member in alliance.members:
                if member in self.country_profiles:
                    profile = self.country_profiles[member]
                    if alliance.leader_country and alliance.leader_country != member:
                        alignment = profile.geopolitical_alignment.get(alliance.leader_country, 0.5)
                        if alignment < 0.4:
                            evolution["membership_changes"].append({
                                "member": member,
                                "action": "potential_exit",
                                "probability": 0.3
                            })
        
        elif scenario == "deepening_cooperation":
            # Successful cooperation strengthens alliance
            stability_increase = 0.05 * years
            new_stability = min(1.0, alliance.stability_score + stability_increase)
            
            evolution["projected_changes"]["stability_score"] = new_stability
            evolution["projected_changes"]["strength"] = self._adjust_alliance_strength(alliance.strength, 0.1)
            
            # Potential for new members
            for country, profile in self.country_profiles.items():
                if country not in alliance.members:
                    avg_alignment = np.mean([profile.geopolitical_alignment.get(member, 0.5) 
                                           for member in alliance.members])
                    if avg_alignment > 0.6:
                        evolution["membership_changes"].append({
                            "member": country,
                            "action": "potential_join",
                            "probability": avg_alignment * 0.5
                        })
        
        elif scenario == "economic_crisis":
            # Economic pressure tests alliance cohesion
            crisis_impact = 0.15
            new_stability = max(0.0, alliance.stability_score - crisis_impact)
            
            evolution["projected_changes"]["stability_score"] = new_stability
            
            # Members may prioritize economic over strategic interests
            for member in alliance.members:
                economic_benefit = alliance.economic_benefits.get(member, 0.5)
                if economic_benefit < 0.4:
                    evolution["membership_changes"].append({
                        "member": member,
                        "action": "reduced_commitment",
                        "probability": 0.4
                    })
        
        return evolution
    
    def _adjust_alliance_strength(self, current_strength: AllianceStrength, adjustment: float) -> AllianceStrength:
        """Adjust alliance strength based on evolution."""
        strength_values = {
            AllianceStrength.WEAK: 1,
            AllianceStrength.MODERATE: 2,
            AllianceStrength.STRONG: 3,
            AllianceStrength.STRATEGIC: 4
        }
        
        reverse_mapping = {v: k for k, v in strength_values.items()}
        
        current_value = strength_values[current_strength]
        new_value = max(1, min(4, current_value + int(adjustment * 10)))
        
        return reverse_mapping[new_value]
    
    def get_alliance_summary(self) -> Dict[str, Any]:
        """Get comprehensive alliance landscape summary."""
        
        # Network metrics
        network_metrics = {
            "total_countries": len(self.country_profiles),
            "active_alliances": len(self.alliances),
            "network_density": nx.density(self.alliance_network),
            "average_clustering": nx.average_clustering(self.alliance_network),
            "connected_components": nx.number_connected_components(self.alliance_network)
        }
        
        # Alliance type distribution
        type_distribution = {}
        for alliance in self.alliances.values():
            alliance_type = alliance.alliance_type.value
            type_distribution[alliance_type] = type_distribution.get(alliance_type, 0) + 1
        
        # Most influential countries (by alliance membership)
        influence_scores = {}
        for country in self.country_profiles.keys():
            alliance_count = sum(1 for alliance in self.alliances.values() if country in alliance.members)
            leadership_count = sum(1 for alliance in self.alliances.values() if alliance.leader_country == country)
            influence_scores[country] = alliance_count + (leadership_count * 2)
        
        # Recent formation predictions
        if not self.alliance_proposals:
            self.predict_alliance_formation()
        
        return {
            "network_overview": network_metrics,
            "alliance_landscape": {
                "by_type": type_distribution,
                "by_strength": {strength.value: sum(1 for a in self.alliances.values() if a.strength == strength) 
                              for strength in AllianceStrength},
                "average_stability": np.mean([a.stability_score for a in self.alliances.values()]),
                "most_exclusive": max(self.alliances.values(), key=lambda x: x.exclusivity_level, default=None)
            },
            "country_influence": dict(sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)),
            "formation_predictions": {
                "total_proposals": len(self.alliance_proposals),
                "high_probability": len([p for p in self.alliance_proposals if p.formation_probability > 0.7]),
                "most_likely": self.alliance_proposals[0] if self.alliance_proposals else None
            },
            "key_alliances": {
                alliance_id: {
                    "type": alliance.alliance_type.value,
                    "members": alliance.members,
                    "leader": alliance.leader_country,
                    "strength": alliance.strength.value,
                    "stability": alliance.stability_score
                }
                for alliance_id, alliance in self.alliances.items()
            }
        } 