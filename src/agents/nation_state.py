"""
Nation State Agent Implementation

Models national governments and their policies affecting semiconductor geopolitics,
including export controls, industrial policy, trade relationships, and strategic positioning.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
sys.path.append('src')

from core.base_agent import BaseAgent, AgentType, AgentMetrics
from config.constants import *

class GeopoliticalStance(Enum):
    """Geopolitical stance types."""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    CONFRONTATIONAL = "confrontational"
    NEUTRAL = "neutral"
    ALLIANCE_BUILDER = "alliance_builder"

class IndustrialStrategy(Enum):
    """Industrial development strategies."""
    SELF_SUFFICIENCY = "self_sufficiency"
    TECHNOLOGICAL_LEADERSHIP = "technological_leadership"
    MARKET_ACCESS = "market_access"
    SUPPLY_CHAIN_RESILIENCE = "supply_chain_resilience"
    INNOVATION_HUB = "innovation_hub"

class PolicyTool(Enum):
    """Policy implementation tools."""
    EXPORT_CONTROLS = "export_controls"
    SUBSIDIES = "subsidies"
    TARIFFS = "tariffs"
    INVESTMENT_RESTRICTIONS = "investment_restrictions"
    TECHNOLOGY_TRANSFER_CONTROLS = "technology_transfer_controls"
    RESEARCH_FUNDING = "research_funding"
    SANCTIONS = "sanctions"

@dataclass
class TradeRelationship:
    """Bilateral trade relationship details."""
    partner_nation: str
    partner_agent_id: Optional[int]
    trade_volume_billions: float
    tech_export_value_billions: float
    relationship_strength: float  # -1 to 1
    trust_level: float  # 0 to 1
    dependency_score: float  # 0 to 1
    strategic_importance: float  # 0 to 1
    sanctions_in_place: List[str]
    export_restrictions: List[str]

@dataclass
class IndustrialPolicy:
    """Industrial policy program details."""
    policy_name: str
    policy_type: PolicyTool
    budget_billions: float
    target_sector: str
    implementation_timeline_months: int
    effectiveness_score: float  # 0 to 1
    domestic_support: float  # 0 to 1
    international_resistance: float  # 0 to 1
    strategic_objectives: List[str]

@dataclass
class TechnologyCapability:
    """National technology capability assessment."""
    sector: str
    capability_level: float  # 0 to 1
    self_sufficiency_ratio: float  # 0 to 1
    innovation_capacity: float  # 0 to 1
    manufacturing_capacity: float  # 0 to 1
    talent_pool_size: int
    r_and_d_investment_billions: float
    time_to_competitiveness_years: int

class NationStateAgent(BaseAgent):
    """
    Nation-state agent representing national governments in semiconductor geopolitics.
    
    Models strategic decisions around:
    - Industrial policy and economic security
    - Export controls and trade restrictions
    - Bilateral and multilateral relationships
    - Technology development and innovation policy
    - Strategic alliance formation
    """
    
    def __init__(self, unique_id: int, model, name: str, **kwargs):
        super().__init__(
            unique_id=unique_id,
            model=model,
            agent_type=AgentType.NATION_STATE,
            name=name,
            **kwargs
        )
        
        # Nation-state specific attributes
        self.country_name = kwargs.get('country_name', name)
        self.gdp_trillions = kwargs.get('gdp_trillions', 1.0)
        self.tech_competitiveness_rank = kwargs.get('tech_competitiveness', 50)
        self.semiconductor_market_share = kwargs.get('semiconductor_share', 0.0)
        self.geopolitical_influence = kwargs.get('geopolitical_influence', 0.5)
        
        # Strategic positioning
        self.geopolitical_stance = GeopoliticalStance.NEUTRAL
        self.industrial_strategy = IndustrialStrategy.SUPPLY_CHAIN_RESILIENCE
        self.strategic_priorities = kwargs.get('priorities', ['economic_security', 'innovation'])
        
        # Policy framework
        self.active_policies: List[IndustrialPolicy] = []
        self.policy_budget_billions = kwargs.get('policy_budget', 10.0)
        self.implementation_capacity = kwargs.get('implementation_capacity', 0.7)
        
        # Relationships and capabilities
        self.trade_relationships: Dict[str, TradeRelationship] = {}
        self.technology_capabilities: Dict[str, TechnologyCapability] = {}
        self.alliance_memberships: List[str] = []
        
        # Performance metrics
        self.policy_effectiveness = 0.6
        self.domestic_approval = 0.7
        self.international_reputation = 0.6
        self.economic_security_index = 0.5
        self.innovation_index = 0.5
        
        # Initialize state
        self._initialize_trade_relationships()
        self._initialize_technology_capabilities()
        self._initialize_policy_framework()
    
    def _initialize_trade_relationships(self):
        """Initialize bilateral trade relationships."""
        # Major trading partners for different countries
        relationship_templates = {
            'United_States': [
                ('China', 600, 20, -0.3, 0.3, 0.4, 0.9, ['entity_list'], ['advanced_semiconductors']),
                ('Taiwan', 80, 15, 0.8, 0.9, 0.3, 0.95, [], []),
                ('South_Korea', 120, 12, 0.7, 0.8, 0.2, 0.8, [], []),
                ('Japan', 150, 10, 0.9, 0.95, 0.1, 0.85, [], []),
                ('Germany', 100, 5, 0.8, 0.9, 0.1, 0.7, [], [])
            ],
            'China': [
                ('United_States', 600, 20, -0.3, 0.3, 0.5, 0.9, [], ['advanced_semiconductors']),
                ('Taiwan', 200, 25, 0.2, 0.6, 0.7, 0.95, [], []),
                ('South_Korea', 250, 18, 0.5, 0.7, 0.4, 0.8, [], []),
                ('Japan', 300, 8, 0.3, 0.5, 0.3, 0.7, [], []),
                ('ASEAN', 400, 15, 0.7, 0.8, 0.2, 0.6, [], [])
            ],
            'European_Union': [
                ('United_States', 800, 15, 0.7, 0.8, 0.2, 0.8, [], []),
                ('China', 700, 10, 0.1, 0.5, 0.3, 0.7, [], ['dual_use_tech']),
                ('Taiwan', 50, 8, 0.6, 0.7, 0.3, 0.8, [], []),
                ('South_Korea', 80, 6, 0.6, 0.7, 0.2, 0.6, [], [])
            ]
        }
        
        # Initialize based on country
        country_key = self.country_name.replace(' ', '_')
        if country_key in relationship_templates:
            for partner, trade_vol, tech_export, rel_strength, trust, dependency, importance, sanctions, restrictions in relationship_templates[country_key]:
                self.trade_relationships[partner] = TradeRelationship(
                    partner_nation=partner,
                    partner_agent_id=None,
                    trade_volume_billions=trade_vol,
                    tech_export_value_billions=tech_export,
                    relationship_strength=rel_strength,
                    trust_level=trust,
                    dependency_score=dependency,
                    strategic_importance=importance,
                    sanctions_in_place=sanctions,
                    export_restrictions=restrictions
                )
    
    def _initialize_technology_capabilities(self):
        """Initialize national technology capability assessment."""
        # Technology sectors and baseline capabilities by country
        sectors = ['chip_design', 'chip_manufacturing', 'equipment_manufacturing', 'materials', 'packaging']
        
        capability_profiles = {
            'United_States': {
                'chip_design': (0.9, 0.8, 0.95, 0.7, 150000, 50, 2),
                'chip_manufacturing': (0.4, 0.3, 0.7, 0.6, 80000, 30, 5),
                'equipment_manufacturing': (0.8, 0.7, 0.9, 0.8, 50000, 25, 3),
                'materials': (0.6, 0.5, 0.7, 0.7, 30000, 15, 4),
                'packaging': (0.5, 0.4, 0.6, 0.7, 40000, 8, 3)
            },
            'China': {
                'chip_design': (0.6, 0.3, 0.7, 0.8, 200000, 40, 3),
                'chip_manufacturing': (0.6, 0.4, 0.6, 0.9, 300000, 60, 4),
                'equipment_manufacturing': (0.3, 0.2, 0.4, 0.7, 100000, 35, 8),
                'materials': (0.4, 0.3, 0.5, 0.8, 80000, 20, 6),
                'packaging': (0.8, 0.7, 0.7, 0.9, 150000, 12, 2)
            },
            'Taiwan': {
                'chip_design': (0.8, 0.6, 0.8, 0.9, 80000, 8, 2),
                'chip_manufacturing': (0.95, 0.9, 0.95, 0.95, 120000, 15, 1),
                'equipment_manufacturing': (0.4, 0.3, 0.5, 0.6, 20000, 3, 5),
                'materials': (0.5, 0.4, 0.6, 0.7, 15000, 2, 4),
                'packaging': (0.9, 0.8, 0.8, 0.9, 60000, 5, 1)
            }
        }
        
        # Default profile for other countries
        default_profile = {
            sector: (0.4, 0.3, 0.5, 0.5, 20000, 5, 6) for sector in sectors
        }
        
        country_key = self.country_name.replace(' ', '_')
        profile = capability_profiles.get(country_key, default_profile)
        
        for sector in sectors:
            if sector in profile:
                capability, self_suff, innovation, manufacturing, talent, rd, time_to_comp = profile[sector]
                self.technology_capabilities[sector] = TechnologyCapability(
                    sector=sector,
                    capability_level=capability,
                    self_sufficiency_ratio=self_suff,
                    innovation_capacity=innovation,
                    manufacturing_capacity=manufacturing,
                    talent_pool_size=talent,
                    r_and_d_investment_billions=rd,
                    time_to_competitiveness_years=time_to_comp
                )
    
    def _initialize_policy_framework(self):
        """Initialize active policy framework."""
        # Country-specific policy templates
        policy_templates = {
            'United_States': [
                ('CHIPS_Act', PolicyTool.SUBSIDIES, 52, 'chip_manufacturing', 60, 0.7, 0.6, 0.3, ['supply_chain_resilience', 'technological_leadership']),
                ('Export_Controls_China', PolicyTool.EXPORT_CONTROLS, 2, 'advanced_semiconductors', 12, 0.8, 0.5, 0.8, ['national_security', 'technological_advantage'])
            ],
            'China': [
                ('National_IC_Plan', PolicyTool.SUBSIDIES, 150, 'semiconductor_industry', 120, 0.6, 0.8, 0.6, ['self_sufficiency', 'technological_independence']),
                ('Big_Fund_III', PolicyTool.RESEARCH_FUNDING, 47, 'chip_design', 84, 0.7, 0.9, 0.2, ['innovation_capacity', 'talent_development'])
            ],
            'European_Union': [
                ('EU_Chips_Act', PolicyTool.SUBSIDIES, 43, 'chip_manufacturing', 72, 0.6, 0.7, 0.4, ['strategic_autonomy', 'supply_chain_resilience']),
                ('Digital_Sovereignty', PolicyTool.INVESTMENT_RESTRICTIONS, 5, 'critical_technologies', 36, 0.5, 0.6, 0.5, ['technological_sovereignty'])
            ]
        }
        
        country_key = self.country_name.replace(' ', '_')
        if country_key in policy_templates:
            for name, tool, budget, sector, timeline, effectiveness, support, resistance, objectives in policy_templates[country_key]:
                self.active_policies.append(IndustrialPolicy(
                    policy_name=name,
                    policy_type=tool,
                    budget_billions=budget,
                    target_sector=sector,
                    implementation_timeline_months=timeline,
                    effectiveness_score=effectiveness,
                    domestic_support=support,
                    international_resistance=resistance,
                    strategic_objectives=objectives
                ))
    
    def step(self):
        """Execute one simulation step."""
        # Assess geopolitical environment
        self._assess_geopolitical_environment()
        
        # Make strategic decisions
        decisions = self.make_strategic_decisions()
        
        # Execute policy implementation
        self._execute_policy_implementation()
        
        # Manage international relationships
        self._manage_international_relationships()
        
        # Update technology capabilities
        self._update_technology_capabilities()
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Track overall metrics
        self.update_metrics()
        
        return decisions
    
    def make_strategic_decisions(self) -> Dict[str, Any]:
        """Make strategic decisions for this time step."""
        decisions = {}
        
        # 1. Policy adjustment decisions
        decisions['policy_adjustments'] = self._decide_policy_adjustments()
        
        # 2. Trade relationship management
        decisions['trade_policy'] = self._decide_trade_policy()
        
        # 3. Technology development priorities
        decisions['technology_policy'] = self._decide_technology_policy()
        
        # 4. Alliance and partnership strategy
        decisions['alliance_strategy'] = self._decide_alliance_strategy()
        
        # 5. Economic security measures
        decisions['economic_security'] = self._decide_economic_security_measures()
        
        return decisions
    
    def _decide_policy_adjustments(self) -> Dict[str, Any]:
        """Decide on policy adjustments and new initiatives."""
        # Assess current policy effectiveness
        policy_performance = {}
        for policy in self.active_policies:
            # Calculate policy performance score
            effectiveness = policy.effectiveness_score
            public_support = policy.domestic_support
            international_cost = policy.international_resistance
            budget_efficiency = min(1.0, policy.budget_billions / 10)  # Normalized
            
            performance_score = (
                effectiveness * 0.4 +
                public_support * 0.3 +
                (1 - international_cost) * 0.2 +
                budget_efficiency * 0.1
            )
            
            policy_performance[policy.policy_name] = {
                'performance_score': performance_score,
                'effectiveness': effectiveness,
                'budget_utilization': policy.budget_billions,
                'timeline_remaining': policy.implementation_timeline_months
            }
        
        # Identify policy gaps
        technology_gaps = self._identify_technology_gaps()
        security_vulnerabilities = self._assess_security_vulnerabilities()
        
        # Recommend new policies
        new_policy_recommendations = []
        available_budget = self.policy_budget_billions * 0.3  # 30% for new initiatives
        
        if technology_gaps['chip_manufacturing'] > 0.5 and available_budget > 10:
            new_policy_recommendations.append({
                'policy_type': PolicyTool.SUBSIDIES.value,
                'target_sector': 'chip_manufacturing',
                'recommended_budget': 20,
                'priority_score': 0.9,
                'timeline_months': 48
            })
        
        if security_vulnerabilities['supply_chain'] > 0.6 and available_budget > 5:
            new_policy_recommendations.append({
                'policy_type': PolicyTool.INVESTMENT_RESTRICTIONS.value,
                'target_sector': 'critical_technologies',
                'recommended_budget': 5,
                'priority_score': 0.8,
                'timeline_months': 24
            })
        
        return {
            'current_policy_performance': policy_performance,
            'technology_gaps': technology_gaps,
            'security_vulnerabilities': security_vulnerabilities,
            'new_policy_recommendations': new_policy_recommendations,
            'available_budget': available_budget
        }
    
    def _decide_trade_policy(self) -> Dict[str, Any]:
        """Decide trade policy and relationship management."""
        trade_actions = []
        
        for partner, relationship in self.trade_relationships.items():
            # Assess relationship dynamics
            economic_importance = relationship.trade_volume_billions / 100  # Normalize
            security_risk = relationship.dependency_score * (1 - relationship.trust_level)
            strategic_value = relationship.strategic_importance
            
            # Calculate action priority
            action_urgency = (
                economic_importance * 0.3 +
                security_risk * 0.4 +
                strategic_value * 0.3
            )
            
            # Determine action type
            if relationship.relationship_strength < -0.5:
                action_type = "escalate_restrictions"
                details = {
                    'new_export_controls': ['advanced_ai_chips', 'quantum_technologies'],
                    'investment_screening': True,
                    'tariff_adjustments': 0.25
                }
            elif relationship.relationship_strength < 0:
                action_type = "defensive_measures"
                details = {
                    'supply_chain_diversification': True,
                    'strategic_stockpiling': True,
                    'technology_protection': True
                }
            elif relationship.relationship_strength > 0.5:
                action_type = "deepen_cooperation"
                details = {
                    'technology_sharing_agreements': True,
                    'joint_rd_programs': True,
                    'trade_facilitation': True
                }
            else:
                action_type = "maintain_status_quo"
                details = {
                    'regular_dialogue': True,
                    'monitoring_enhanced': True
                }
            
            trade_actions.append({
                'partner': partner,
                'action_type': action_type,
                'action_urgency': action_urgency,
                'current_relationship': relationship.relationship_strength,
                'details': details
            })
        
        # Multilateral trade strategy
        multilateral_priorities = self._assess_multilateral_opportunities()
        
        return {
            'bilateral_actions': trade_actions,
            'multilateral_priorities': multilateral_priorities,
            'trade_diversification_targets': self._identify_diversification_targets()
        }
    
    def _decide_technology_policy(self) -> Dict[str, Any]:
        """Decide technology development and innovation policy."""
        # Assess technology development priorities
        technology_priorities = {}
        total_rd_budget = sum(cap.r_and_d_investment_billions for cap in self.technology_capabilities.values())
        
        for sector, capability in self.technology_capabilities.items():
            # Priority scoring
            capability_gap = 1.0 - capability.capability_level
            self_sufficiency_gap = 1.0 - capability.self_sufficiency_ratio
            strategic_importance = 0.9 if sector in ['chip_manufacturing', 'chip_design'] else 0.6
            time_sensitivity = 1.0 / max(1, capability.time_to_competitiveness_years)
            
            priority_score = (
                capability_gap * 0.3 +
                self_sufficiency_gap * 0.25 +
                strategic_importance * 0.25 +
                time_sensitivity * 0.2
            )
            
            technology_priorities[sector] = {
                'priority_score': priority_score,
                'capability_gap': capability_gap,
                'self_sufficiency_gap': self_sufficiency_gap,
                'recommended_investment': total_rd_budget * priority_score,
                'timeline_years': capability.time_to_competitiveness_years
            }
        
        # International collaboration opportunities
        collaboration_opportunities = self._identify_collaboration_opportunities()
        
        # Talent development strategy
        talent_strategy = self._develop_talent_strategy()
        
        return {
            'technology_priorities': technology_priorities,
            'collaboration_opportunities': collaboration_opportunities,
            'talent_strategy': talent_strategy,
            'total_rd_budget': total_rd_budget
        }
    
    def _decide_alliance_strategy(self) -> Dict[str, Any]:
        """Decide alliance formation and partnership strategy."""
        # Assess potential alliance partners
        potential_partners = {}
        
        for partner, relationship in self.trade_relationships.items():
            # Alliance compatibility score
            trust_factor = relationship.trust_level
            strategic_alignment = relationship.strategic_importance
            mutual_benefit = min(relationship.dependency_score, 0.5) * 2  # Mutual dependency is good for alliances
            geopolitical_compatibility = 1.0 if relationship.relationship_strength > 0 else 0.2
            
            compatibility_score = (
                trust_factor * 0.3 +
                strategic_alignment * 0.3 +
                mutual_benefit * 0.2 +
                geopolitical_compatibility * 0.2
            )
            
            potential_partners[partner] = {
                'compatibility_score': compatibility_score,
                'alliance_type': self._determine_alliance_type(compatibility_score),
                'potential_benefits': self._assess_alliance_benefits(partner),
                'risks': self._assess_alliance_risks(partner)
            }
        
        # Multilateral organization strategy
        multilateral_strategy = {
            'semiconductor_alliance': {
                'participation_level': 'active' if self.geopolitical_stance in [GeopoliticalStance.ALLIANCE_BUILDER, GeopoliticalStance.COOPERATIVE] else 'observer',
                'leadership_ambition': 0.8 if self.geopolitical_influence > 0.7 else 0.3,
                'resource_commitment': self.policy_budget_billions * 0.1
            },
            'trade_agreements': {
                'new_agreements_priority': ['CPTPP', 'Digital_Economy_Partnership'],
                'existing_agreement_depth': 'enhance' if self.industrial_strategy == IndustrialStrategy.MARKET_ACCESS else 'maintain'
            }
        }
        
        return {
            'potential_partners': potential_partners,
            'multilateral_strategy': multilateral_strategy,
            'alliance_priorities': self._rank_alliance_priorities()
        }
    
    def _decide_economic_security_measures(self) -> Dict[str, Any]:
        """Decide economic security and resilience measures."""
        # Assess current vulnerabilities
        vulnerabilities = self._assess_economic_vulnerabilities()
        
        # Resilience measures
        resilience_measures = {
            'strategic_stockpiles': {
                'target_sectors': ['critical_materials', 'advanced_semiconductors'],
                'stockpile_duration_months': 6,
                'investment_required': vulnerabilities['supply_chain'] * 10  # billions
            },
            'domestic_capacity_building': {
                'priority_sectors': ['chip_manufacturing', 'equipment_manufacturing'],
                'capacity_targets': {
                    'chip_manufacturing': min(0.8, self.technology_capabilities['chip_manufacturing'].self_sufficiency_ratio + 0.2),
                    'equipment_manufacturing': min(0.6, self.technology_capabilities['equipment_manufacturing'].self_sufficiency_ratio + 0.15)
                },
                'timeline_years': 5
            },
            'supply_chain_diversification': {
                'diversification_targets': self._identify_diversification_targets(),
                'risk_reduction_goal': 0.3,
                'investment_incentives': True
            }
        }
        
        # Monitoring and intelligence
        intelligence_priorities = {
            'technology_monitoring': ['quantum_computing', 'ai_accelerators', 'advanced_materials'],
            'supply_chain_monitoring': ['critical_chokepoints', 'alternative_suppliers'],
            'competitor_analysis': ['technology_capabilities', 'industrial_policies', 'alliance_formations']
        }
        
        return {
            'vulnerabilities': vulnerabilities,
            'resilience_measures': resilience_measures,
            'intelligence_priorities': intelligence_priorities,
            'total_security_investment': sum(measure.get('investment_required', 0) for measure in resilience_measures.values())
        }
    
    def assess_geopolitical_risk(self) -> float:
        """Assess current geopolitical risk exposure."""
        # Base risk from global tensions
        base_risk = 0.3
        
        # Trade relationship risks
        relationship_risk = 0
        for relationship in self.trade_relationships.values():
            partner_risk = max(0, -relationship.relationship_strength) * relationship.strategic_importance
            relationship_risk += partner_risk
        relationship_risk /= len(self.trade_relationships) if self.trade_relationships else 1
        
        # Technology dependency risk
        tech_risk = 0
        for capability in self.technology_capabilities.values():
            dependency_risk = (1 - capability.self_sufficiency_ratio) * 0.1
            tech_risk += dependency_risk
        tech_risk /= len(self.technology_capabilities) if self.technology_capabilities else 1
        
        # Policy implementation risk
        policy_risk = (1 - self.implementation_capacity) * 0.2
        
        total_risk = base_risk + relationship_risk * 0.4 + tech_risk * 0.3 + policy_risk
        return np.clip(total_risk, 0.0, 1.0)
    
    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get summary of current strategic position."""
        return {
            'agent_id': self.unique_id,
            'country_name': self.country_name,
            'gdp_trillions': self.gdp_trillions,
            'geopolitical_stance': self.geopolitical_stance.value,
            'industrial_strategy': self.industrial_strategy.value,
            'tech_competitiveness_rank': self.tech_competitiveness_rank,
            'semiconductor_market_share': self.semiconductor_market_share,
            'geopolitical_influence': self.geopolitical_influence,
            'active_policies': len(self.active_policies),
            'policy_budget': self.policy_budget_billions,
            'trade_partners': len(self.trade_relationships),
            'alliance_memberships': self.alliance_memberships,
            'technology_strengths': self._identify_technology_strengths(),
            'technology_gaps': self._identify_technology_gaps(),
            'geopolitical_risk': self.assess_geopolitical_risk(),
            'economic_security_index': self.economic_security_index,
            'innovation_index': self.innovation_index
        }
    
    # Helper methods for analysis and assessment
    def _assess_geopolitical_environment(self):
        """Assess current geopolitical environment."""
        # Update stance based on relationship dynamics
        avg_relationship_strength = np.mean([r.relationship_strength for r in self.trade_relationships.values()])
        
        if avg_relationship_strength > 0.5:
            self.geopolitical_stance = GeopoliticalStance.COOPERATIVE
        elif avg_relationship_strength < -0.3:
            self.geopolitical_stance = GeopoliticalStance.CONFRONTATIONAL
        else:
            self.geopolitical_stance = GeopoliticalStance.COMPETITIVE
    
    def _identify_technology_gaps(self) -> Dict[str, float]:
        """Identify technology capability gaps."""
        gaps = {}
        for sector, capability in self.technology_capabilities.items():
            gaps[sector] = max(0, 0.8 - capability.capability_level)  # Target 80% capability
        return gaps
    
    def _assess_security_vulnerabilities(self) -> Dict[str, float]:
        """Assess economic and security vulnerabilities."""
        vulnerabilities = {}
        
        # Supply chain vulnerability
        supply_chain_risk = np.mean([1 - cap.self_sufficiency_ratio for cap in self.technology_capabilities.values()])
        vulnerabilities['supply_chain'] = supply_chain_risk
        
        # Trade dependency risk
        trade_risk = np.mean([rel.dependency_score for rel in self.trade_relationships.values()])
        vulnerabilities['trade_dependency'] = trade_risk
        
        # Technology dependency
        tech_dependency = 1 - np.mean([cap.capability_level for cap in self.technology_capabilities.values()])
        vulnerabilities['technology_dependency'] = tech_dependency
        
        return vulnerabilities
    
    def _identify_technology_strengths(self) -> List[str]:
        """Identify national technology strengths."""
        strengths = []
        for sector, capability in self.technology_capabilities.items():
            if capability.capability_level > 0.7:
                strengths.append(sector)
        return strengths
    
    def _assess_multilateral_opportunities(self) -> List[Dict]:
        """Assess multilateral cooperation opportunities."""
        return [
            {'organization': 'Semiconductor_Alliance', 'priority': 0.8, 'resource_requirement': 5},
            {'organization': 'Critical_Materials_Partnership', 'priority': 0.6, 'resource_requirement': 3},
            {'organization': 'AI_Governance_Framework', 'priority': 0.7, 'resource_requirement': 2}
        ]
    
    def _identify_diversification_targets(self) -> List[str]:
        """Identify supply chain diversification targets."""
        return ['Southeast_Asia', 'Latin_America', 'Eastern_Europe', 'India']
    
    def _identify_collaboration_opportunities(self) -> List[Dict]:
        """Identify international collaboration opportunities."""
        return [
            {'type': 'joint_rd', 'partners': ['allied_nations'], 'investment': 5, 'timeline': 36},
            {'type': 'talent_exchange', 'partners': ['technology_leaders'], 'investment': 2, 'timeline': 12},
            {'type': 'standard_setting', 'partners': ['industry_leaders'], 'investment': 1, 'timeline': 24}
        ]
    
    def _develop_talent_strategy(self) -> Dict[str, Any]:
        """Develop talent development strategy."""
        return {
            'domestic_education': {'investment': 10, 'target_graduates': 50000, 'timeline': 48},
            'international_recruitment': {'investment': 3, 'target_professionals': 10000, 'timeline': 24},
            'industry_partnerships': {'investment': 5, 'program_count': 20, 'timeline': 36}
        }
    
    def _determine_alliance_type(self, compatibility_score: float) -> str:
        """Determine appropriate alliance type based on compatibility."""
        if compatibility_score > 0.8:
            return "strategic_partnership"
        elif compatibility_score > 0.6:
            return "technology_cooperation"
        elif compatibility_score > 0.4:
            return "trade_partnership"
        else:
            return "limited_engagement"
    
    def _assess_alliance_benefits(self, partner: str) -> List[str]:
        """Assess potential benefits of alliance with partner."""
        return ["technology_sharing", "market_access", "supply_chain_resilience", "cost_sharing"]
    
    def _assess_alliance_risks(self, partner: str) -> List[str]:
        """Assess potential risks of alliance with partner."""
        return ["technology_leakage", "dependency_creation", "geopolitical_entanglement"]
    
    def _rank_alliance_priorities(self) -> List[str]:
        """Rank alliance formation priorities."""
        return ["technology_cooperation", "supply_chain_security", "market_access", "standard_setting"]
    
    def _assess_economic_vulnerabilities(self) -> Dict[str, float]:
        """Assess economic vulnerabilities."""
        return {
            'supply_chain': 0.6,
            'technology_dependency': 0.5,
            'trade_concentration': 0.4,
            'critical_materials': 0.7
        }
    
    # Implementation and update methods
    def _execute_policy_implementation(self):
        """Execute policy implementation and track progress."""
        for policy in self.active_policies:
            policy.implementation_timeline_months -= 1
            
            # Update effectiveness based on implementation progress
            if policy.implementation_timeline_months > 0:
                progress_factor = 1 - (policy.implementation_timeline_months / 120)  # Assume max 10 years
                policy.effectiveness_score = min(1.0, policy.effectiveness_score + progress_factor * 0.01)
    
    def _manage_international_relationships(self):
        """Manage and update international relationships."""
        for relationship in self.trade_relationships.values():
            # Relationship evolution based on policies and interactions
            policy_impact = self._calculate_policy_impact_on_relationship(relationship)
            random_events = self.random.normal(0, 0.02)
            
            relationship.relationship_strength = np.clip(
                relationship.relationship_strength + policy_impact + random_events, -1.0, 1.0
            )
            
            relationship.trust_level = np.clip(
                relationship.trust_level + (policy_impact * 0.5) + random_events, 0.0, 1.0
            )
    
    def _calculate_policy_impact_on_relationship(self, relationship: TradeRelationship) -> float:
        """Calculate impact of current policies on relationship."""
        impact = 0
        for policy in self.active_policies:
            if policy.policy_type == PolicyTool.EXPORT_CONTROLS:
                impact -= 0.05  # Negative impact
            elif policy.policy_type == PolicyTool.SUBSIDIES:
                impact -= 0.02  # Slight negative impact on competitors
            elif policy.policy_type == PolicyTool.RESEARCH_FUNDING:
                impact += 0.01  # Slight positive impact
        return np.clip(impact, -0.1, 0.1)
    
    def _update_technology_capabilities(self):
        """Update national technology capabilities."""
        for capability in self.technology_capabilities.values():
            # Investment-driven capability improvement
            investment_impact = capability.r_and_d_investment_billions / 50  # Normalize
            capability.capability_level = min(1.0, capability.capability_level + investment_impact * 0.002)
            
            # Self-sufficiency improvement through industrial policy
            policy_impact = sum(policy.effectiveness_score for policy in self.active_policies 
                              if policy.target_sector == capability.sector) / 100
            capability.self_sufficiency_ratio = min(1.0, capability.self_sufficiency_ratio + policy_impact)
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        # Policy effectiveness
        if self.active_policies:
            self.policy_effectiveness = np.mean([p.effectiveness_score for p in self.active_policies])
        
        # Economic security index
        tech_security = np.mean([cap.self_sufficiency_ratio for cap in self.technology_capabilities.values()])
        trade_security = 1 - np.mean([rel.dependency_score for rel in self.trade_relationships.values()])
        self.economic_security_index = (tech_security + trade_security) / 2
        
        # Innovation index
        innovation_investment = sum(cap.r_and_d_investment_billions for cap in self.technology_capabilities.values())
        self.innovation_index = min(1.0, innovation_investment / (self.gdp_trillions * 100))  # Normalize 