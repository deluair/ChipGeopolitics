"""
Chip Manufacturer Agent Implementation

Models semiconductor foundries and IDMs (TSMC, Samsung, Intel, GlobalFoundries, etc.)
with strategic decision-making around capacity allocation, technology roadmaps,
and geopolitical positioning.
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

class ManufacturingStrategy(Enum):
    """Manufacturing strategy types."""
    CUTTING_EDGE_LEADER = "cutting_edge_leader"
    VOLUME_OPTIMIZATION = "volume_optimization"
    SPECIALIZED_NODES = "specialized_nodes"
    COST_LEADERSHIP = "cost_leadership"
    GEOGRAPHIC_HEDGING = "geographic_hedging"

class TechnologyFocus(Enum):
    """Technology development focus areas."""
    LOGIC_ADVANCEMENT = "logic_advancement"
    MEMORY_INTEGRATION = "memory_integration"
    POWER_EFFICIENCY = "power_efficiency"
    AI_OPTIMIZATION = "ai_optimization"
    AUTOMOTIVE_GRADE = "automotive_grade"

@dataclass
class ProcessNode:
    """Process node technology details."""
    node_size: str  # e.g., "5nm", "3nm"
    maturity_level: float  # 0-1, where 1 is fully mature
    yield_rate: float  # 0-1
    cost_per_wafer: float  # USD
    development_cost_billions: float
    time_to_market_months: int
    equipment_requirements: List[str]
    customer_demand: float  # Relative demand score

@dataclass
class FabCapacity:
    """Fabrication facility capacity details."""
    fab_name: str
    location: str
    node_capabilities: List[str]
    monthly_wafer_capacity: int
    utilization_rate: float
    equipment_vintage: float  # Years
    geopolitical_risk: float
    expansion_potential_wafers: int

@dataclass
class CustomerContract:
    """Customer contract details."""
    customer_name: str
    customer_agent_id: Optional[int]
    contract_value_billions: float
    wafer_allocation: int
    preferred_nodes: List[str]
    contract_duration_months: int
    renewal_probability: float
    pricing_premium: float  # Multiplier above baseline

class ChipManufacturerAgent(BaseAgent):
    """
    Chip manufacturer agent representing semiconductor foundries and IDMs.
    
    Models strategic decisions around:
    - Process technology roadmap development
    - Capacity allocation and fab expansion
    - Customer relationship management
    - Geographic risk management
    - Technology licensing and partnerships
    """
    
    def __init__(self, unique_id: int, model, name: str, **kwargs):
        super().__init__(
            unique_id=unique_id,
            model=model,
            agent_type=AgentType.CHIP_MANUFACTURER,
            name=name,
            **kwargs
        )
        
        # Chip manufacturer specific attributes
        self.market_share = kwargs.get('market_share', 0.0)
        self.monthly_capacity_wafers = kwargs.get('monthly_capacity', 0)
        self.technology_leadership_score = kwargs.get('technology_leadership', 0.5)
        self.process_nodes = kwargs.get('process_nodes', ['28nm'])
        self.primary_region = kwargs.get('primary_region', 'Asia Pacific')
        
        # Strategic positioning
        self.manufacturing_strategy = ManufacturingStrategy.VOLUME_OPTIMIZATION
        self.technology_focus = TechnologyFocus.LOGIC_ADVANCEMENT
        self.geographic_hedge_ratio = kwargs.get('geographic_hedge_ratio', 0.1)
        
        # Technology portfolio
        self.node_portfolio: Dict[str, ProcessNode] = {}
        self.fab_portfolio: List[FabCapacity] = []
        self.customer_contracts: Dict[str, CustomerContract] = {}
        
        # Performance metrics
        self.average_yield_rate = 0.75
        self.average_utilization = 0.85
        self.rd_intensity = kwargs.get('rd_intensity', 0.15)  # R&D as % of revenue
        self.customer_satisfaction = 0.8
        
        # Technology roadmap
        self.roadmap_investments: Dict[str, float] = {}
        self.partnership_agreements: List[str] = []
        
        # Initialize capabilities
        self._initialize_process_nodes()
        self._initialize_fab_capacity()
        self._initialize_customer_base()
    
    def _initialize_process_nodes(self):
        """Initialize process node capabilities based on current industry state."""
        # Define available process nodes with realistic parameters
        node_configs = {
            '28nm': (0.95, 0.9, 2500, 0.05, 6, ['litho_193i', 'etch_multi'], 0.6),
            '16nm': (0.9, 0.85, 4000, 0.1, 12, ['litho_193i_multi', 'etch_advanced'], 0.7),
            '10nm': (0.85, 0.8, 6000, 0.3, 18, ['euv_limited', 'etch_extreme'], 0.8),
            '7nm': (0.8, 0.75, 8500, 0.5, 24, ['euv_multi', 'etch_extreme'], 0.85),
            '5nm': (0.75, 0.7, 9566, 0.7, 30, ['euv_intensive', 'etch_next_gen'], 0.9),
            '3nm': (0.6, 0.65, 12000, 1.0, 36, ['euv_advanced', 'gate_all_around'], 0.95),
            '2nm': (0.3, 0.5, 17000, 2.0, 48, ['euv_high_na', 'nanosheet'], 1.0)
        }
        
        # Initialize nodes based on agent's capabilities
        for node in self.process_nodes:
            if node in node_configs:
                config = node_configs[node]
                self.node_portfolio[node] = ProcessNode(
                    node_size=node,
                    maturity_level=config[0],
                    yield_rate=config[1],
                    cost_per_wafer=config[2],
                    development_cost_billions=config[3],
                    time_to_market_months=config[4],
                    equipment_requirements=config[5],
                    customer_demand=config[6]
                )
    
    def _initialize_fab_capacity(self):
        """Initialize fabrication capacity based on company profile."""
        # Create representative fabs
        total_capacity = self.monthly_capacity_wafers
        
        # Distribute capacity across fabs (realistic distribution)
        fab_configs = [
            ("Fab1", self.primary_region, 0.4),
            ("Fab2", self.primary_region, 0.3),
            ("Fab3", "Alternative Region", 0.2),
            ("Fab4", "Backup Region", 0.1)
        ]
        
        for fab_name, location, capacity_ratio in fab_configs:
            fab_capacity = int(total_capacity * capacity_ratio)
            if fab_capacity > 0:
                # Determine geopolitical risk based on location
                geo_risk = 0.1  # Default
                if "Asia Pacific" in location:
                    geo_risk = 0.3
                elif "Alternative" in location:
                    geo_risk = 0.2
                
                self.fab_portfolio.append(FabCapacity(
                    fab_name=fab_name,
                    location=location,
                    node_capabilities=list(self.process_nodes),
                    monthly_wafer_capacity=fab_capacity,
                    utilization_rate=self.average_utilization,
                    equipment_vintage=self.random.uniform(2, 8),  # Years
                    geopolitical_risk=geo_risk,
                    expansion_potential_wafers=fab_capacity // 2
                ))
    
    def _initialize_customer_base(self):
        """Initialize customer contracts based on market position."""
        # Major customer types for chip manufacturers
        customer_types = [
            ("Hyperscaler_A", 0.25, ["5nm", "7nm"], 12, 0.8, 1.2),
            ("Mobile_OEM_A", 0.2, ["7nm", "10nm"], 18, 0.7, 1.1),
            ("Enterprise_A", 0.15, ["16nm", "28nm"], 24, 0.9, 1.0),
            ("Automotive_A", 0.1, ["28nm", "16nm"], 36, 0.95, 1.15),
            ("Various_SMEs", 0.3, ["28nm"], 12, 0.6, 0.9)
        ]
        
        total_revenue = self.monthly_capacity_wafers * 8000 * 12 / 1e9  # Rough revenue estimate
        
        for name, share, nodes, duration, renewal, premium in customer_types:
            contract_value = total_revenue * share
            wafer_allocation = int(self.monthly_capacity_wafers * share)
            
            self.customer_contracts[name] = CustomerContract(
                customer_name=name,
                customer_agent_id=None,  # Will be linked during simulation
                contract_value_billions=contract_value,
                wafer_allocation=wafer_allocation,
                preferred_nodes=nodes,
                contract_duration_months=duration,
                renewal_probability=renewal,
                pricing_premium=premium
            )
    
    def step(self):
        """Execute one simulation step."""
        # Assess market and technology trends
        self._assess_technology_trends()
        
        # Make strategic decisions
        decisions = self.make_strategic_decisions()
        
        # Execute capacity planning
        self._execute_capacity_planning()
        
        # Manage customer relationships
        self._manage_customer_relationships()
        
        # Update technology roadmap
        self._update_technology_roadmap()
        
        # Update financial performance
        self._update_financial_performance()
        
        # Track performance metrics
        self.update_metrics()
        
        return decisions
    
    def make_strategic_decisions(self) -> Dict[str, Any]:
        """Make strategic decisions for this time step."""
        decisions = {}
        
        # 1. Technology roadmap prioritization
        decisions['technology_roadmap'] = self._decide_technology_priorities()
        
        # 2. Capacity allocation optimization
        decisions['capacity_allocation'] = self._decide_capacity_allocation()
        
        # 3. Geographic expansion/hedging
        decisions['geographic_strategy'] = self._decide_geographic_expansion()
        
        # 4. Customer portfolio management
        decisions['customer_strategy'] = self._decide_customer_strategy()
        
        # 5. Technology partnership decisions
        decisions['partnerships'] = self._decide_partnerships()
        
        return decisions
    
    def _decide_technology_priorities(self) -> Dict[str, Any]:
        """Decide technology development priorities and R&D allocation."""
        rd_budget = self.calculate_rd_budget()
        
        # Assess technology needs based on customer demand and competition
        technology_priorities = {}
        
        for node, process_node in self.node_portfolio.items():
            # Priority score based on multiple factors
            demand_score = process_node.customer_demand
            maturity_gap = 1.0 - process_node.maturity_level
            competitive_pressure = self._assess_competitive_pressure(node)
            market_potential = self._estimate_market_potential(node)
            
            priority_score = (
                demand_score * 0.3 +
                maturity_gap * 0.25 +
                competitive_pressure * 0.25 +
                market_potential * 0.2
            )
            
            technology_priorities[node] = {
                'priority_score': priority_score,
                'recommended_investment': rd_budget * priority_score,
                'development_timeline': process_node.time_to_market_months,
                'risk_level': maturity_gap
            }
        
        # Allocate additional budget for next-generation nodes
        next_gen_investment = rd_budget * 0.3
        next_gen_nodes = self._identify_next_generation_nodes()
        
        return {
            'rd_budget_total': rd_budget,
            'current_node_priorities': technology_priorities,
            'next_generation_investment': next_gen_investment,
            'target_nodes': next_gen_nodes,
            'technology_focus': self.technology_focus.value
        }
    
    def _decide_capacity_allocation(self) -> Dict[str, Any]:
        """Decide how to allocate capacity across customers and nodes."""
        # Calculate total available capacity
        total_capacity = sum(fab.monthly_wafer_capacity * fab.utilization_rate 
                           for fab in self.fab_portfolio)
        
        # Assess demand vs capacity
        total_demand = sum(contract.wafer_allocation 
                          for contract in self.customer_contracts.values())
        
        capacity_utilization = total_demand / total_capacity if total_capacity > 0 else 1.0
        
        # Allocation strategy based on utilization
        if capacity_utilization > 0.95:
            strategy = "capacity_constrained"
            # Prioritize high-value customers
            allocation_decisions = self._prioritize_high_value_customers()
        elif capacity_utilization > 0.8:
            strategy = "optimal_utilization"
            # Balanced allocation
            allocation_decisions = self._balanced_capacity_allocation()
        else:
            strategy = "demand_building"
            # Focus on customer acquisition
            allocation_decisions = self._customer_acquisition_allocation()
        
        # Capacity expansion decisions
        expansion_needed = max(0, total_demand - total_capacity)
        expansion_plans = self._plan_capacity_expansion(expansion_needed)
        
        return {
            'strategy': strategy,
            'current_utilization': capacity_utilization,
            'total_capacity_wafers': total_capacity,
            'total_demand_wafers': total_demand,
            'allocation_decisions': allocation_decisions,
            'expansion_plans': expansion_plans
        }
    
    def _decide_geographic_expansion(self) -> Dict[str, Any]:
        """Decide on geographic expansion and risk hedging."""
        # Assess current geographic concentration
        regional_exposure = self._calculate_regional_exposure()
        geopolitical_risk = self.assess_geopolitical_risk()
        
        # Determine expansion priorities
        expansion_priorities = {}
        target_regions = ['North America', 'Europe', 'Southeast Asia', 'Latin America']
        
        for region in target_regions:
            if region not in regional_exposure:
                # New market opportunity
                market_potential = self.random.uniform(0.3, 0.8)
                regulatory_ease = self.random.uniform(0.2, 0.9)
                cost_competitiveness = self.random.uniform(0.1, 0.7)
                
                # Adjust for known regional characteristics
                if region == 'North America':
                    regulatory_ease *= 0.7  # CHIPS Act benefits but complex regulations
                    cost_competitiveness *= 0.5  # Higher costs
                    market_potential *= 1.2  # Strong demand
                elif region == 'Europe':
                    regulatory_ease *= 0.6  # EU sovereignty requirements
                    cost_competitiveness *= 0.6  # High costs
                    market_potential *= 0.9  # Moderate demand
                
                expansion_score = (
                    market_potential * 0.4 +
                    regulatory_ease * 0.3 +
                    cost_competitiveness * 0.3
                )
                
                expansion_priorities[region] = {
                    'expansion_score': expansion_score,
                    'investment_required_billions': self.random.uniform(5, 20),
                    'timeline_months': self.random.randint(24, 60),
                    'risk_level': 1 - regulatory_ease
                }
        
        # Hedging strategy
        hedging_recommendation = "increase_hedging" if geopolitical_risk > 0.5 else "maintain_current"
        
        return {
            'current_exposure': regional_exposure,
            'geopolitical_risk': geopolitical_risk,
            'expansion_priorities': expansion_priorities,
            'hedging_recommendation': hedging_recommendation,
            'target_hedge_ratio': min(0.4, self.geographic_hedge_ratio + 0.1)
        }
    
    def _decide_customer_strategy(self) -> Dict[str, Any]:
        """Decide customer portfolio management strategy."""
        customer_actions = []
        
        for customer_name, contract in self.customer_contracts.items():
            # Assess customer relationship health
            relationship_score = (
                contract.renewal_probability * 0.4 +
                contract.pricing_premium * 0.3 +
                (contract.contract_value_billions / 5) * 0.3  # Value score
            )
            
            # Decision logic
            if relationship_score > 0.8:
                action = "strengthen"
                details = {
                    'action': 'strengthen',
                    'customer': customer_name,
                    'recommended_investment': contract.contract_value_billions * 0.05,
                    'priority_level': 'high'
                }
            elif relationship_score > 0.5:
                action = "maintain"
                details = {
                    'action': 'maintain',
                    'customer': customer_name,
                    'recommended_investment': contract.contract_value_billions * 0.02,
                    'priority_level': 'medium'
                }
            elif relationship_score > 0.3:
                action = "improve"
                details = {
                    'action': 'improve',
                    'customer': customer_name,
                    'recommended_investment': contract.contract_value_billions * 0.08,
                    'priority_level': 'medium'
                }
            else:
                action = "diversify_away"
                details = {
                    'action': 'diversify_away',
                    'customer': customer_name,
                    'recommended_investment': 0,
                    'priority_level': 'low'
                }
            
            customer_actions.append(details)
        
        # Customer acquisition strategy
        acquisition_budget = self.calculate_rd_budget() * 0.1  # 10% of R&D for customer acquisition
        target_segments = self._identify_target_customer_segments()
        
        return {
            'customer_actions': customer_actions,
            'acquisition_budget': acquisition_budget,
            'target_segments': target_segments,
            'portfolio_concentration': self._calculate_customer_concentration()
        }
    
    def _decide_partnerships(self) -> Dict[str, Any]:
        """Decide on technology partnerships and licensing."""
        # Assess partnership opportunities
        partnership_opportunities = {
            'equipment_suppliers': {
                'ASML': {'priority': 0.9, 'investment': 0.5, 'strategic_value': 0.95},
                'Applied_Materials': {'priority': 0.7, 'investment': 0.3, 'strategic_value': 0.8},
                'Tokyo_Electron': {'priority': 0.6, 'investment': 0.2, 'strategic_value': 0.7}
            },
            'research_institutes': {
                'IMEC': {'priority': 0.8, 'investment': 0.1, 'strategic_value': 0.85},
                'CEA_LETI': {'priority': 0.6, 'investment': 0.1, 'strategic_value': 0.7},
                'AIST': {'priority': 0.5, 'investment': 0.05, 'strategic_value': 0.6}
            },
            'technology_companies': {
                'ARM': {'priority': 0.7, 'investment': 0.2, 'strategic_value': 0.8},
                'Cadence': {'priority': 0.6, 'investment': 0.15, 'strategic_value': 0.75},
                'Synopsys': {'priority': 0.6, 'investment': 0.15, 'strategic_value': 0.75}
            }
        }
        
        # Partnership budget allocation
        partnership_budget = self.calculate_rd_budget() * 0.2  # 20% of R&D for partnerships
        
        # Select partnerships based on strategic priorities
        selected_partnerships = []
        remaining_budget = partnership_budget
        
        for category, opportunities in partnership_opportunities.items():
            for partner, details in opportunities.items():
                cost = details['investment'] * partnership_budget
                if (remaining_budget >= cost and 
                    details['priority'] > 0.7 and
                    partner not in self.partnership_agreements):
                    
                    selected_partnerships.append({
                        'partner': partner,
                        'category': category,
                        'investment': cost,
                        'strategic_value': details['strategic_value'],
                        'timeline_months': self.random.randint(12, 36)
                    })
                    remaining_budget -= cost
        
        return {
            'selected_partnerships': selected_partnerships,
            'total_investment': partnership_budget - remaining_budget,
            'existing_partnerships': list(self.partnership_agreements),
            'strategic_focus': self.technology_focus.value
        }
    
    def calculate_rd_budget(self) -> float:
        """Calculate R&D budget based on revenue and strategy."""
        # Estimate annual revenue
        annual_revenue = 0
        for contract in self.customer_contracts.values():
            annual_revenue += contract.contract_value_billions
        
        # R&D budget as percentage of revenue
        rd_budget = annual_revenue * self.rd_intensity
        
        # Adjust based on strategy
        if self.manufacturing_strategy == ManufacturingStrategy.CUTTING_EDGE_LEADER:
            rd_budget *= 1.5
        elif self.manufacturing_strategy == ManufacturingStrategy.COST_LEADERSHIP:
            rd_budget *= 0.7
        
        return rd_budget
    
    def _assess_competitive_pressure(self, node: str) -> float:
        """Assess competitive pressure for a specific process node."""
        # Simplified competitive assessment
        # In a full simulation, this would analyze other manufacturer agents
        
        if node in ['3nm', '2nm']:
            return 0.9  # High competition for cutting-edge
        elif node in ['5nm', '7nm']:
            return 0.7  # Moderate competition
        else:
            return 0.4  # Lower competition for mature nodes
    
    def _estimate_market_potential(self, node: str) -> float:
        """Estimate market potential for a process node."""
        # Market potential based on industry trends
        node_potential = {
            '2nm': 0.95,  # Highest potential
            '3nm': 0.9,
            '5nm': 0.8,
            '7nm': 0.7,
            '10nm': 0.5,
            '16nm': 0.4,
            '28nm': 0.6   # Stable demand for mature node
        }
        
        return node_potential.get(node, 0.3)
    
    def _identify_next_generation_nodes(self) -> List[str]:
        """Identify next-generation nodes to develop."""
        current_best = min(self.process_nodes, key=lambda x: int(x.replace('nm', '')))
        current_nm = int(current_best.replace('nm', ''))
        
        # Next generation targets
        if current_nm <= 2:
            return ['1.4nm', 'angstrom_scale']
        elif current_nm <= 3:
            return ['2nm', '1.4nm']
        elif current_nm <= 5:
            return ['3nm', '2nm']
        else:
            return ['5nm', '3nm']
    
    def _calculate_regional_exposure(self) -> Dict[str, float]:
        """Calculate exposure to different geographic regions."""
        regional_capacity = {}
        total_capacity = sum(fab.monthly_wafer_capacity for fab in self.fab_portfolio)
        
        for fab in self.fab_portfolio:
            region = fab.location
            if region not in regional_capacity:
                regional_capacity[region] = 0
            regional_capacity[region] += fab.monthly_wafer_capacity
        
        # Convert to percentages
        for region in regional_capacity:
            regional_capacity[region] /= total_capacity
        
        return regional_capacity
    
    def _calculate_customer_concentration(self) -> float:
        """Calculate customer concentration using Herfindahl index."""
        total_value = sum(c.contract_value_billions for c in self.customer_contracts.values())
        if total_value == 0:
            return 1.0
        
        shares = [c.contract_value_billions / total_value for c in self.customer_contracts.values()]
        hhi = sum(share ** 2 for share in shares)
        return hhi
    
    def assess_geopolitical_risk(self) -> float:
        """Assess current geopolitical risk exposure."""
        # Base risk from industry conditions
        base_risk = 0.2
        
        # Geographic concentration risk
        regional_exposure = self._calculate_regional_exposure()
        concentration_risk = max(regional_exposure.values()) * 0.3  # Max exposure * factor
        
        # Customer concentration risk
        customer_risk = self._calculate_customer_concentration() * 0.2
        
        # Technology dependency risk (simplified)
        tech_risk = 0.1  # Base technology risk
        
        total_risk = base_risk + concentration_risk + customer_risk + tech_risk
        return np.clip(total_risk, 0.0, 1.0)
    
    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get summary of current strategic position."""
        return {
            'agent_id': self.unique_id,
            'name': self.name,
            'market_share': self.market_share,
            'monthly_capacity': self.monthly_capacity_wafers,
            'technology_leadership': self.technology_leadership_score,
            'process_nodes': list(self.node_portfolio.keys()),
            'manufacturing_strategy': self.manufacturing_strategy.value,
            'technology_focus': self.technology_focus.value,
            'fab_count': len(self.fab_portfolio),
            'customer_count': len(self.customer_contracts),
            'average_utilization': self.average_utilization,
            'rd_intensity': self.rd_intensity,
            'geopolitical_risk': self.assess_geopolitical_risk(),
            'regional_exposure': self._calculate_regional_exposure(),
            'customer_concentration': self._calculate_customer_concentration()
        }
    
    # Additional helper methods for capacity planning and relationship management
    def _assess_technology_trends(self):
        """Assess current technology trends and market dynamics."""
        # Update process node maturity and demand
        for node, process_node in self.node_portfolio.items():
            # Gradual maturity improvement
            process_node.maturity_level = min(1.0, process_node.maturity_level + 0.01)
            
            # Demand evolution
            demand_change = self.random.normal(0, 0.02)
            process_node.customer_demand = np.clip(
                process_node.customer_demand + demand_change, 0.1, 1.0
            )
    
    def _execute_capacity_planning(self):
        """Execute capacity planning decisions."""
        # Update fab utilization based on demand
        for fab in self.fab_portfolio:
            # Adjust utilization based on market conditions
            utilization_change = self.random.normal(0, 0.02)
            fab.utilization_rate = np.clip(
                fab.utilization_rate + utilization_change, 0.5, 0.98
            )
    
    def _manage_customer_relationships(self):
        """Manage and update customer relationships."""
        for customer_name, contract in self.customer_contracts.items():
            # Update renewal probability based on performance
            performance_factor = self.random.normal(0, 0.05)
            contract.renewal_probability = np.clip(
                contract.renewal_probability + performance_factor, 0.1, 0.98
            )
    
    def _update_technology_roadmap(self):
        """Update technology development roadmap."""
        # Progress on current developments
        for node in self.roadmap_investments:
            self.roadmap_investments[node] *= 1.1  # Compound investment
    
    def _update_financial_performance(self):
        """Update financial performance metrics."""
        # Calculate monthly revenue from contracts
        monthly_revenue = sum(c.contract_value_billions / 12 for c in self.customer_contracts.values())
        
        # Calculate costs (simplified)
        monthly_costs = monthly_revenue * 0.75  # 75% cost ratio
        
        # Update history
        self.revenue_history.append(monthly_revenue)
        self.cost_history.append(monthly_costs)
        
        # Update capital
        self.capital += (monthly_revenue - monthly_costs)
    
    # Simplified helper methods for capacity allocation decisions
    def _prioritize_high_value_customers(self) -> List[Dict]:
        return [{'strategy': 'prioritize_high_value', 'details': 'Focus on premium customers'}]
    
    def _balanced_capacity_allocation(self) -> List[Dict]:
        return [{'strategy': 'balanced_allocation', 'details': 'Optimize across all customers'}]
    
    def _customer_acquisition_allocation(self) -> List[Dict]:
        return [{'strategy': 'customer_acquisition', 'details': 'Reserve capacity for new customers'}]
    
    def _plan_capacity_expansion(self, expansion_needed: float) -> List[Dict]:
        if expansion_needed > 0:
            return [{
                'expansion_type': 'new_fab_line',
                'capacity_increase': expansion_needed,
                'estimated_cost': expansion_needed * 0.002,  # $2M per wafer capacity
                'timeline_months': 18
            }]
        return []
    
    def _identify_target_customer_segments(self) -> List[str]:
        return ['AI_Companies', 'Automotive_OEMs', 'IoT_Manufacturers', 'Telecom_Equipment'] 