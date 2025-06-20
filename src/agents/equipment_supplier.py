"""
Equipment Supplier Agent Implementation

Models semiconductor equipment suppliers (ASML, Applied Materials, Tokyo Electron, etc.)
with strategic decision-making around technology development, market positioning,
and geopolitical navigation.
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

class EquipmentCategory(Enum):
    """Equipment category types."""
    LITHOGRAPHY = "lithography"
    ETCH = "etch"
    DEPOSITION = "deposition"
    CMP = "cmp"  # Chemical Mechanical Planarization
    METROLOGY = "metrology"
    ION_IMPLANTATION = "ion_implantation"
    THERMAL = "thermal"

class TechnologyGenerations(Enum):
    """Technology generation levels."""
    MATURE = "mature"  # 28nm+
    ADVANCED = "advanced"  # 7-16nm
    LEADING_EDGE = "leading_edge"  # 3-5nm
    NEXT_GEN = "next_gen"  # 2nm+

class MarketStrategy(Enum):
    """Market positioning strategies."""
    TECHNOLOGY_LEADER = "technology_leader"
    COST_OPTIMIZER = "cost_optimizer"
    MARKET_FOLLOWER = "market_follower"
    NICHE_SPECIALIST = "niche_specialist"

@dataclass
class EquipmentProduct:
    """Equipment product details."""
    product_name: str
    category: EquipmentCategory
    technology_generation: TechnologyGenerations
    market_share: float
    unit_price_millions: float
    development_cost_millions: float
    time_to_market_months: int
    competitive_advantage: float  # 0-1
    geopolitical_sensitivity: float  # 0-1

@dataclass
class CustomerRelationship:
    """Equipment supplier customer relationship."""
    customer_name: str
    customer_agent_id: Optional[int]
    relationship_strength: float
    annual_revenue_millions: float
    equipment_categories: List[EquipmentCategory]
    contract_duration_months: int
    technology_collaboration: float  # 0-1
    geographic_restrictions: List[str]

@dataclass
class RDProject:
    """R&D project details."""
    project_name: str
    category: EquipmentCategory
    target_generation: TechnologyGenerations
    budget_millions: float
    timeline_months: int
    completion_probability: float
    strategic_importance: float
    collaboration_partners: List[str]

class EquipmentSupplierAgent(BaseAgent):
    """
    Equipment supplier agent representing semiconductor equipment manufacturers.
    
    Models strategic decisions around:
    - Technology roadmap and R&D investment
    - Product portfolio optimization
    - Customer relationship management
    - Geopolitical risk navigation
    - Market positioning and competitive strategy
    """
    
    def __init__(self, unique_id: int, model, name: str, **kwargs):
        super().__init__(
            unique_id=unique_id,
            model=model,
            agent_type=AgentType.EQUIPMENT_SUPPLIER,
            name=name,
            **kwargs
        )
        
        # Equipment supplier specific attributes
        self.equipment_categories = kwargs.get('equipment_categories', [EquipmentCategory.LITHOGRAPHY])
        self.market_position = kwargs.get('market_position', MarketStrategy.MARKET_FOLLOWER)
        self.technology_leadership = kwargs.get('technology_leadership', 0.5)
        self.global_market_share = kwargs.get('global_market_share', 0.0)
        
        # Strategic focus
        self.innovation_focus = kwargs.get('innovation_focus', TechnologyGenerations.ADVANCED)
        self.geographic_coverage = kwargs.get('geographic_coverage', ['Asia Pacific'])
        self.rd_intensity = kwargs.get('rd_intensity', 0.20)  # R&D as % of revenue
        
        # Product portfolio
        self.product_portfolio: Dict[str, EquipmentProduct] = {}
        self.customer_relationships: Dict[str, CustomerRelationship] = {}
        self.rd_projects: List[RDProject] = []
        
        # Performance metrics
        self.innovation_pipeline_value = 0.0
        self.customer_satisfaction = 0.8
        self.technology_advantage_score = 0.5
        self.manufacturing_efficiency = 0.7
        
        # Market dynamics
        self.competitive_pressure = 0.5
        self.regulatory_compliance_cost = 0.0
        self.export_restriction_impact = 0.0
        
        # Initialize portfolio and relationships
        self._initialize_product_portfolio()
        self._initialize_customer_relationships()
        self._initialize_rd_projects()
    
    def _initialize_product_portfolio(self):
        """Initialize product portfolio based on equipment categories."""
        # Define product configurations by category
        product_configs = {
            EquipmentCategory.LITHOGRAPHY: [
                ("EUV_Scanner", TechnologyGenerations.LEADING_EDGE, 0.7, 200, 500, 48, 0.9, 0.95),
                ("ArF_Immersion", TechnologyGenerations.ADVANCED, 0.4, 80, 100, 24, 0.6, 0.3),
                ("i_line_Stepper", TechnologyGenerations.MATURE, 0.2, 5, 20, 12, 0.3, 0.1)
            ],
            EquipmentCategory.ETCH: [
                ("Plasma_Etch_Advanced", TechnologyGenerations.LEADING_EDGE, 0.3, 15, 50, 18, 0.7, 0.4),
                ("RIE_System", TechnologyGenerations.ADVANCED, 0.4, 8, 25, 12, 0.5, 0.2),
                ("Wet_Etch", TechnologyGenerations.MATURE, 0.1, 2, 5, 6, 0.2, 0.1)
            ],
            EquipmentCategory.DEPOSITION: [
                ("ALD_System", TechnologyGenerations.LEADING_EDGE, 0.4, 12, 40, 15, 0.6, 0.3),
                ("CVD_Chamber", TechnologyGenerations.ADVANCED, 0.3, 6, 20, 10, 0.4, 0.2),
                ("PVD_Sputter", TechnologyGenerations.MATURE, 0.2, 3, 8, 8, 0.3, 0.1)
            ],
            EquipmentCategory.CMP: [
                ("CMP_Polisher", TechnologyGenerations.ADVANCED, 0.25, 4, 15, 12, 0.5, 0.2)
            ],
            EquipmentCategory.METROLOGY: [
                ("Overlay_Metrology", TechnologyGenerations.LEADING_EDGE, 0.3, 8, 25, 15, 0.6, 0.4),
                ("CD_SEM", TechnologyGenerations.ADVANCED, 0.2, 3, 10, 10, 0.4, 0.2)
            ]
        }
        
        # Initialize products for agent's categories
        for category in self.equipment_categories:
            if category in product_configs:
                for config in product_configs[category]:
                    name, gen, share, price, dev_cost, ttm, advantage, geo_sens = config
                    
                    # Adjust based on agent's market position
                    if self.market_position == MarketStrategy.TECHNOLOGY_LEADER:
                        share *= 1.5
                        advantage *= 1.2
                        price *= 1.1
                    elif self.market_position == MarketStrategy.COST_OPTIMIZER:
                        price *= 0.8
                        dev_cost *= 0.7
                        advantage *= 0.8
                    
                    self.product_portfolio[name] = EquipmentProduct(
                        product_name=name,
                        category=category,
                        technology_generation=gen,
                        market_share=min(1.0, share),
                        unit_price_millions=price,
                        development_cost_millions=dev_cost,
                        time_to_market_months=ttm,
                        competitive_advantage=min(1.0, advantage),
                        geopolitical_sensitivity=geo_sens
                    )
    
    def _initialize_customer_relationships(self):
        """Initialize customer relationships based on market position."""
        # Major customer types for equipment suppliers
        customer_types = [
            ("TSMC", 0.8, 500, [EquipmentCategory.LITHOGRAPHY, EquipmentCategory.ETCH], 36, 0.9, []),
            ("Samsung", 0.7, 300, [EquipmentCategory.LITHOGRAPHY, EquipmentCategory.DEPOSITION], 24, 0.8, []),
            ("Intel", 0.6, 200, [EquipmentCategory.METROLOGY, EquipmentCategory.CMP], 24, 0.7, []),
            ("GlobalFoundries", 0.5, 100, [EquipmentCategory.ETCH, EquipmentCategory.DEPOSITION], 18, 0.6, []),
            ("SMIC", 0.4, 80, [EquipmentCategory.DEPOSITION], 12, 0.5, ["US_restricted"]),
            ("Regional_Fabs", 0.3, 150, [EquipmentCategory.CMP, EquipmentCategory.METROLOGY], 12, 0.4, [])
        ]
        
        for name, strength, revenue, categories, duration, collab, restrictions in customer_types:
            # Adjust relationship strength based on market position
            if self.market_position == MarketStrategy.TECHNOLOGY_LEADER:
                strength *= 1.2
                revenue *= 1.3
            elif self.market_position == MarketStrategy.COST_OPTIMIZER:
                revenue *= 0.8
            
            # Filter categories based on agent's capabilities
            relevant_categories = [cat for cat in categories if cat in self.equipment_categories]
            
            if relevant_categories:  # Only add if we serve this customer
                self.customer_relationships[name] = CustomerRelationship(
                    customer_name=name,
                    customer_agent_id=None,
                    relationship_strength=min(1.0, strength),
                    annual_revenue_millions=revenue,
                    equipment_categories=relevant_categories,
                    contract_duration_months=duration,
                    technology_collaboration=collab,
                    geographic_restrictions=restrictions
                )
    
    def _initialize_rd_projects(self):
        """Initialize R&D project pipeline."""
        # R&D projects based on technology focus
        project_templates = {
            TechnologyGenerations.NEXT_GEN: [
                ("High_NA_EUV", EquipmentCategory.LITHOGRAPHY, 800, 60, 0.7, 0.95),
                ("Atomic_Layer_Etch", EquipmentCategory.ETCH, 200, 36, 0.8, 0.85),
                ("Quantum_Metrology", EquipmentCategory.METROLOGY, 150, 48, 0.6, 0.9)
            ],
            TechnologyGenerations.LEADING_EDGE: [
                ("EUV_Pellicle", EquipmentCategory.LITHOGRAPHY, 300, 24, 0.8, 0.8),
                ("Plasma_Damage_Free", EquipmentCategory.ETCH, 100, 18, 0.7, 0.7),
                ("AI_Process_Control", EquipmentCategory.METROLOGY, 80, 15, 0.9, 0.6)
            ],
            TechnologyGenerations.ADVANCED: [
                ("Multi_Beam_Litho", EquipmentCategory.LITHOGRAPHY, 150, 18, 0.8, 0.5),
                ("Selective_Deposition", EquipmentCategory.DEPOSITION, 75, 12, 0.7, 0.4)
            ]
        }
        
        # Add projects based on innovation focus
        focus_projects = project_templates.get(self.innovation_focus, [])
        for name, category, budget, timeline, probability, importance in focus_projects:
            if category in self.equipment_categories:
                self.rd_projects.append(RDProject(
                    project_name=name,
                    category=category,
                    target_generation=self.innovation_focus,
                    budget_millions=budget,
                    timeline_months=timeline,
                    completion_probability=probability,
                    strategic_importance=importance,
                    collaboration_partners=[]
                ))
    
    def step(self):
        """Execute one simulation step."""
        # Assess market conditions
        self._assess_market_dynamics()
        
        # Make strategic decisions
        decisions = self.make_strategic_decisions()
        
        # Execute R&D projects
        self._execute_rd_projects()
        
        # Manage customer relationships
        self._manage_customer_relationships()
        
        # Update product portfolio
        self._update_product_portfolio()
        
        # Update financial performance
        self._update_financial_performance()
        
        # Track performance metrics
        self.update_metrics()
        
        return decisions
    
    def make_strategic_decisions(self) -> Dict[str, Any]:
        """Make strategic decisions for this time step."""
        decisions = {}
        
        # 1. R&D investment allocation
        decisions['rd_allocation'] = self._decide_rd_allocation()
        
        # 2. Market positioning strategy
        decisions['market_strategy'] = self._decide_market_strategy()
        
        # 3. Customer portfolio management
        decisions['customer_management'] = self._decide_customer_strategy()
        
        # 4. Geographic expansion
        decisions['geographic_strategy'] = self._decide_geographic_strategy()
        
        # 5. Technology partnerships
        decisions['partnerships'] = self._decide_partnership_strategy()
        
        return decisions
    
    def _decide_rd_allocation(self) -> Dict[str, Any]:
        """Decide R&D budget allocation across projects and categories."""
        total_rd_budget = self.calculate_rd_budget()
        
        # Assess project priorities
        project_priorities = {}
        for project in self.rd_projects:
            # Priority scoring
            strategic_value = project.strategic_importance
            market_potential = self._estimate_market_potential(project.category, project.target_generation)
            competitive_urgency = self._assess_competitive_pressure(project.category)
            success_probability = project.completion_probability
            
            priority_score = (
                strategic_value * 0.3 +
                market_potential * 0.25 +
                competitive_urgency * 0.25 +
                success_probability * 0.2
            )
            
            project_priorities[project.project_name] = {
                'priority_score': priority_score,
                'recommended_budget': min(project.budget_millions, total_rd_budget * priority_score),
                'timeline': project.timeline_months,
                'risk_level': 1 - success_probability
            }
        
        # Category-level allocation
        category_allocation = {}
        for category in self.equipment_categories:
            category_projects = [p for p in self.rd_projects if p.category == category]
            category_budget = sum(project_priorities[p.project_name]['recommended_budget'] 
                                for p in category_projects)
            category_allocation[category.value] = category_budget
        
        # Future technology scouting budget
        scouting_budget = total_rd_budget * 0.15
        
        return {
            'total_budget': total_rd_budget,
            'project_priorities': project_priorities,
            'category_allocation': category_allocation,
            'scouting_budget': scouting_budget,
            'innovation_focus': self.innovation_focus.value
        }
    
    def _decide_market_strategy(self) -> Dict[str, Any]:
        """Decide market positioning and competitive strategy."""
        # Assess current market position
        market_conditions = self._analyze_market_conditions()
        
        # Evaluate strategy options
        strategy_options = {
            MarketStrategy.TECHNOLOGY_LEADER: {
                'investment_required': self.calculate_rd_budget() * 1.5,
                'risk_level': 0.7,
                'potential_return': 1.8,
                'time_horizon': 36
            },
            MarketStrategy.COST_OPTIMIZER: {
                'investment_required': self.calculate_rd_budget() * 0.8,
                'risk_level': 0.3,
                'potential_return': 1.2,
                'time_horizon': 12
            },
            MarketStrategy.NICHE_SPECIALIST: {
                'investment_required': self.calculate_rd_budget() * 1.0,
                'risk_level': 0.5,
                'potential_return': 1.5,
                'time_horizon': 24
            }
        }
        
        # Select optimal strategy based on capabilities and market conditions
        current_capability = self.technology_leadership
        market_volatility = market_conditions.get('volatility', 0.5)
        competitive_intensity = market_conditions.get('competitive_intensity', 0.5)
        
        if current_capability > 0.8 and market_volatility < 0.4:
            recommended_strategy = MarketStrategy.TECHNOLOGY_LEADER
        elif competitive_intensity > 0.7:
            recommended_strategy = MarketStrategy.COST_OPTIMIZER
        else:
            recommended_strategy = MarketStrategy.NICHE_SPECIALIST
        
        return {
            'current_strategy': self.market_position.value,
            'recommended_strategy': recommended_strategy.value,
            'strategy_options': strategy_options,
            'market_conditions': market_conditions,
            'transition_timeline': 18  # months
        }
    
    def _decide_customer_strategy(self) -> Dict[str, Any]:
        """Decide customer portfolio management strategy."""
        customer_actions = []
        
        # Analyze each customer relationship
        for customer_name, relationship in self.customer_relationships.items():
            # Calculate customer value score
            revenue_importance = relationship.annual_revenue_millions / 100  # Normalize
            strategic_value = relationship.technology_collaboration
            relationship_health = relationship.relationship_strength
            geo_risk = len(relationship.geographic_restrictions) * 0.2
            
            customer_score = (
                revenue_importance * 0.3 +
                strategic_value * 0.3 +
                relationship_health * 0.25 +
                (1 - geo_risk) * 0.15
            )
            
            # Determine action based on score
            if customer_score > 0.8:
                action_type = "strategic_partnership"
                investment = relationship.annual_revenue_millions * 0.08
            elif customer_score > 0.6:
                action_type = "strengthen_relationship"
                investment = relationship.annual_revenue_millions * 0.05
            elif customer_score > 0.4:
                action_type = "maintain_current"
                investment = relationship.annual_revenue_millions * 0.02
            else:
                action_type = "evaluate_exit"
                investment = 0
            
            customer_actions.append({
                'customer': customer_name,
                'action': action_type,
                'customer_score': customer_score,
                'recommended_investment': investment,
                'geographic_restrictions': relationship.geographic_restrictions
            })
        
        # Customer acquisition strategy
        acquisition_targets = self._identify_acquisition_targets()
        acquisition_budget = self.calculate_rd_budget() * 0.1
        
        return {
            'customer_actions': customer_actions,
            'acquisition_targets': acquisition_targets,
            'acquisition_budget': acquisition_budget,
            'portfolio_diversification': self._calculate_customer_diversification()
        }
    
    def _decide_geographic_strategy(self) -> Dict[str, Any]:
        """Decide geographic expansion and risk management strategy."""
        # Assess current geographic exposure
        current_exposure = self._calculate_geographic_exposure()
        geopolitical_risk = self.assess_geopolitical_risk()
        
        # Evaluate expansion opportunities
        expansion_opportunities = {
            'North_America': {
                'market_potential': 0.7,
                'regulatory_complexity': 0.6,
                'competitive_intensity': 0.8,
                'investment_required': 200  # millions
            },
            'Europe': {
                'market_potential': 0.5,
                'regulatory_complexity': 0.7,
                'competitive_intensity': 0.6,
                'investment_required': 150
            },
            'Southeast_Asia': {
                'market_potential': 0.8,
                'regulatory_complexity': 0.4,
                'competitive_intensity': 0.5,
                'investment_required': 100
            }
        }
        
        # Risk mitigation strategy
        if geopolitical_risk > 0.6:
            risk_strategy = "aggressive_diversification"
            target_hedge_ratio = 0.4
        elif geopolitical_risk > 0.4:
            risk_strategy = "moderate_hedging"
            target_hedge_ratio = 0.25
        else:
            risk_strategy = "maintain_focus"
            target_hedge_ratio = 0.15
        
        return {
            'current_exposure': current_exposure,
            'geopolitical_risk': geopolitical_risk,
            'expansion_opportunities': expansion_opportunities,
            'risk_strategy': risk_strategy,
            'target_hedge_ratio': target_hedge_ratio
        }
    
    def _decide_partnership_strategy(self) -> Dict[str, Any]:
        """Decide technology partnership and collaboration strategy."""
        # Partnership categories
        partnership_types = {
            'research_institutes': {
                'strategic_value': 0.8,
                'cost': 20,  # millions
                'timeline': 36,
                'risk': 0.3
            },
            'complementary_suppliers': {
                'strategic_value': 0.6,
                'cost': 50,
                'timeline': 24,
                'risk': 0.4
            },
            'customer_codevelopment': {
                'strategic_value': 0.9,
                'cost': 100,
                'timeline': 18,
                'risk': 0.2
            },
            'technology_licensing': {
                'strategic_value': 0.5,
                'cost': 30,
                'timeline': 12,
                'risk': 0.5
            }
        }
        
        # Budget allocation
        partnership_budget = self.calculate_rd_budget() * 0.25
        
        # Select partnerships based on strategy and budget
        selected_partnerships = []
        remaining_budget = partnership_budget
        
        for partnership_type, details in partnership_types.items():
            if (remaining_budget >= details['cost'] and 
                details['strategic_value'] > 0.7):
                
                selected_partnerships.append({
                    'type': partnership_type,
                    'investment': details['cost'],
                    'strategic_value': details['strategic_value'],
                    'timeline': details['timeline'],
                    'risk_level': details['risk']
                })
                remaining_budget -= details['cost']
        
        return {
            'selected_partnerships': selected_partnerships,
            'total_investment': partnership_budget - remaining_budget,
            'partnership_budget': partnership_budget,
            'current_partnerships': len(self.rd_projects)  # Simplified
        }
    
    def calculate_rd_budget(self) -> float:
        """Calculate R&D budget based on revenue and strategy."""
        # Estimate annual revenue from customer relationships
        annual_revenue = sum(rel.annual_revenue_millions for rel in self.customer_relationships.values())
        
        # R&D budget as percentage of revenue
        rd_budget = annual_revenue * self.rd_intensity
        
        # Adjust based on market strategy
        if self.market_position == MarketStrategy.TECHNOLOGY_LEADER:
            rd_budget *= 1.4
        elif self.market_position == MarketStrategy.COST_OPTIMIZER:
            rd_budget *= 0.8
        elif self.market_position == MarketStrategy.NICHE_SPECIALIST:
            rd_budget *= 1.1
        
        return rd_budget
    
    def _estimate_market_potential(self, category: EquipmentCategory, generation: TechnologyGenerations) -> float:
        """Estimate market potential for equipment category and generation."""
        # Base market potential by category
        category_potential = {
            EquipmentCategory.LITHOGRAPHY: 0.9,
            EquipmentCategory.ETCH: 0.7,
            EquipmentCategory.DEPOSITION: 0.6,
            EquipmentCategory.METROLOGY: 0.5,
            EquipmentCategory.CMP: 0.4,
            EquipmentCategory.ION_IMPLANTATION: 0.4,
            EquipmentCategory.THERMAL: 0.3
        }
        
        # Generation multipliers
        generation_multiplier = {
            TechnologyGenerations.NEXT_GEN: 1.2,
            TechnologyGenerations.LEADING_EDGE: 1.0,
            TechnologyGenerations.ADVANCED: 0.7,
            TechnologyGenerations.MATURE: 0.4
        }
        
        base_potential = category_potential.get(category, 0.5)
        gen_multiplier = generation_multiplier.get(generation, 0.7)
        
        return min(1.0, base_potential * gen_multiplier)
    
    def _assess_competitive_pressure(self, category: EquipmentCategory) -> float:
        """Assess competitive pressure in equipment category."""
        # Simplified competitive assessment
        # In full simulation, would analyze other equipment supplier agents
        
        category_competition = {
            EquipmentCategory.LITHOGRAPHY: 0.9,  # ASML dominance but high stakes
            EquipmentCategory.ETCH: 0.7,
            EquipmentCategory.DEPOSITION: 0.6,
            EquipmentCategory.METROLOGY: 0.5,
            EquipmentCategory.CMP: 0.4,
            EquipmentCategory.ION_IMPLANTATION: 0.5,
            EquipmentCategory.THERMAL: 0.3
        }
        
        return category_competition.get(category, 0.5)
    
    def assess_geopolitical_risk(self) -> float:
        """Assess current geopolitical risk exposure."""
        # Base risk from global tensions
        base_risk = 0.3
        
        # Product sensitivity risk
        sensitivity_risk = 0
        for product in self.product_portfolio.values():
            sensitivity_risk += product.geopolitical_sensitivity * product.market_share
        sensitivity_risk /= len(self.product_portfolio) if self.product_portfolio else 1
        
        # Customer geographic risk
        customer_risk = 0
        for relationship in self.customer_relationships.values():
            geo_restrictions = len(relationship.geographic_restrictions) * 0.1
            customer_risk += geo_restrictions * (relationship.annual_revenue_millions / 100)
        customer_risk /= len(self.customer_relationships) if self.customer_relationships else 1
        
        # Technology export control risk
        export_risk = 0.2 if any(cat in [EquipmentCategory.LITHOGRAPHY, EquipmentCategory.METROLOGY] 
                               for cat in self.equipment_categories) else 0.1
        
        total_risk = base_risk + sensitivity_risk * 0.3 + customer_risk * 0.3 + export_risk
        return np.clip(total_risk, 0.0, 1.0)
    
    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get summary of current strategic position."""
        return {
            'agent_id': self.unique_id,
            'name': self.name,
            'equipment_categories': [cat.value for cat in self.equipment_categories],
            'market_position': self.market_position.value,
            'technology_leadership': self.technology_leadership,
            'global_market_share': self.global_market_share,
            'innovation_focus': self.innovation_focus.value,
            'product_count': len(self.product_portfolio),
            'customer_count': len(self.customer_relationships),
            'rd_projects': len(self.rd_projects),
            'rd_intensity': self.rd_intensity,
            'geopolitical_risk': self.assess_geopolitical_risk(),
            'customer_diversification': self._calculate_customer_diversification(),
            'geographic_coverage': self.geographic_coverage
        }
    
    # Helper methods for analysis and execution
    def _analyze_market_conditions(self) -> Dict[str, float]:
        """Analyze current market conditions."""
        return {
            'volatility': self.random.uniform(0.2, 0.8),
            'competitive_intensity': self.random.uniform(0.3, 0.9),
            'demand_growth': self.random.uniform(-0.1, 0.4),
            'technology_disruption_risk': self.random.uniform(0.1, 0.6)
        }
    
    def _calculate_customer_diversification(self) -> float:
        """Calculate customer diversification using Herfindahl index."""
        if not self.customer_relationships:
            return 1.0
        
        total_revenue = sum(r.annual_revenue_millions for r in self.customer_relationships.values())
        if total_revenue == 0:
            return 1.0
        
        shares = [r.annual_revenue_millions / total_revenue for r in self.customer_relationships.values()]
        hhi = sum(share ** 2 for share in shares)
        return hhi
    
    def _calculate_geographic_exposure(self) -> Dict[str, float]:
        """Calculate geographic exposure distribution."""
        # Simplified based on customer locations
        exposure = {}
        total_revenue = sum(r.annual_revenue_millions for r in self.customer_relationships.values())
        
        for customer_name, relationship in self.customer_relationships.items():
            # Simplified geographic mapping
            if "TSMC" in customer_name or "SMIC" in customer_name:
                region = "Asia_Pacific"
            elif "Intel" in customer_name:
                region = "North_America"
            elif "Global" in customer_name:
                region = "Europe"
            else:
                region = "Other"
            
            if region not in exposure:
                exposure[region] = 0
            exposure[region] += relationship.annual_revenue_millions / total_revenue if total_revenue > 0 else 0
        
        return exposure
    
    def _identify_acquisition_targets(self) -> List[str]:
        """Identify potential customer acquisition targets."""
        return [
            "Emerging_Foundries",
            "IDM_Expansion",
            "Automotive_Fabs",
            "Power_Semiconductor_Manufacturers"
        ]
    
    def _assess_market_dynamics(self):
        """Assess and update market dynamics."""
        # Update competitive pressure
        self.competitive_pressure = np.clip(
            self.competitive_pressure + self.random.normal(0, 0.05), 0.0, 1.0
        )
        
        # Update technology advantage
        tech_progress = 0.01 if self.market_position == MarketStrategy.TECHNOLOGY_LEADER else 0.005
        self.technology_advantage_score = min(1.0, self.technology_advantage_score + tech_progress)
    
    def _execute_rd_projects(self):
        """Execute and update R&D projects."""
        completed_projects = []
        for project in self.rd_projects:
            project.timeline_months -= 1
            if project.timeline_months <= 0:
                # Project completed
                if self.random.random() < project.completion_probability:
                    # Successful completion
                    self.innovation_pipeline_value += project.budget_millions * 1.5
                    self.technology_leadership = min(1.0, self.technology_leadership + 0.05)
                completed_projects.append(project)
        
        # Remove completed projects
        for project in completed_projects:
            self.rd_projects.remove(project)
    
    def _manage_customer_relationships(self):
        """Manage and update customer relationships."""
        for relationship in self.customer_relationships.values():
            # Relationship evolution
            performance_impact = self.random.normal(0, 0.02)
            relationship.relationship_strength = np.clip(
                relationship.relationship_strength + performance_impact, 0.1, 1.0
            )
    
    def _update_product_portfolio(self):
        """Update product portfolio based on market feedback."""
        for product in self.product_portfolio.values():
            # Market share evolution
            share_change = self.random.normal(0, 0.01)
            product.market_share = np.clip(product.market_share + share_change, 0.01, 1.0)
    
    def _update_financial_performance(self):
        """Update financial performance metrics."""
        # Calculate monthly revenue
        monthly_revenue = sum(r.annual_revenue_millions / 12 for r in self.customer_relationships.values())
        
        # Calculate costs
        rd_costs = self.calculate_rd_budget() / 12
        operational_costs = monthly_revenue * 0.6  # 60% cost ratio
        total_costs = rd_costs + operational_costs
        
        # Update history
        self.revenue_history.append(monthly_revenue)
        self.cost_history.append(total_costs)
        
        # Update capital
        self.capital += (monthly_revenue - total_costs) 