"""
Hyperscaler Agent Implementation

Models cloud service providers (AWS, Google Cloud, Microsoft Azure, etc.)
with strategic decision-making around data center expansion, chip procurement,
and supply chain diversification.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Add project root to path for imports
sys.path.append('src')

from core.base_agent import BaseAgent, AgentType, AgentMetrics
from config.constants import *

class DataCenterStrategy(Enum):
    """Data center expansion strategies."""
    AGGRESSIVE_GROWTH = "aggressive_growth"
    MODERATE_EXPANSION = "moderate_expansion"
    CONSOLIDATION = "consolidation"
    GEOGRAPHIC_DIVERSIFICATION = "geographic_diversification"

class ChipProcurementStrategy(Enum):
    """Chip procurement strategies."""
    SINGLE_SOURCE = "single_source"
    DUAL_SOURCE = "dual_source"
    MULTI_SOURCE = "multi_source"
    VERTICAL_INTEGRATION = "vertical_integration"

@dataclass
class DataCenterPlan:
    """Data center expansion plan."""
    region: str
    planned_capacity_mw: float
    estimated_cost_billions: float
    timeline_months: int
    regulatory_risk_score: float
    grid_availability: float

@dataclass
class SupplierRelationship:
    """Supplier relationship details."""
    supplier_name: str
    agent_id: Optional[int]
    relationship_strength: float  # 0-1
    contract_value_billions: float
    dependency_score: float  # 0-1, higher = more dependent
    alternative_sources: int
    geopolitical_risk: float  # 0-1

class HyperscalerAgent(BaseAgent):
    """
    Hyperscaler agent representing cloud service providers.
    
    Models strategic decisions around:
    - Data center capacity planning and geographic expansion
    - Chip procurement and supplier diversification
    - Technology investment and R&D allocation
    - Geopolitical risk management
    """
    
    def __init__(self, unique_id: int, model, name: str, **kwargs):
        super().__init__(
            unique_id=unique_id,
            model=model,
            agent_type=AgentType.HYPERSCALER,
            name=name,
            **kwargs
        )
        
        # Hyperscaler-specific attributes
        self.cloud_market_share = kwargs.get('cloud_market_share', 0.0)
        self.data_center_count = kwargs.get('data_center_count', 0)
        self.annual_capex = kwargs.get('annual_capex', 0.0)
        self.chip_procurement_volume = kwargs.get('chip_procurement_volume', 0)
        self.ai_workload_percentage = kwargs.get('ai_workload_percentage', 0.3)
        
        # Strategic preferences
        self.risk_tolerance = kwargs.get('risk_tolerance', 0.5)
        self.expansion_aggressiveness = kwargs.get('expansion_aggressiveness', 0.5)
        self.sustainability_commitment = kwargs.get('sustainability_commitment', 0.5)
        
        # Decision state
        self.current_dc_strategy = DataCenterStrategy.MODERATE_EXPANSION
        self.current_procurement_strategy = ChipProcurementStrategy.DUAL_SOURCE
        
        # Planning and relationships
        self.dc_expansion_plans: List[DataCenterPlan] = []
        self.supplier_relationships: Dict[str, SupplierRelationship] = {}
        self.technology_investments: Dict[str, float] = {
            'quantum_computing': 0.0,
            'neuromorphic_chips': 0.0,
            'optical_computing': 0.0,
            'edge_infrastructure': 0.0
        }
        
        # Performance tracking
        self.capacity_utilization = 0.75  # 75% baseline
        self.energy_efficiency_score = 0.6  # PUE-based score
        self.customer_satisfaction = 0.8
        self.innovation_index = 0.5
        
        # Initialize with realistic relationships
        self._initialize_supplier_relationships()
    
    def _initialize_supplier_relationships(self):
        """Initialize realistic supplier relationships based on market data."""
        # Major chip suppliers for hyperscalers
        suppliers = [
            ('NVIDIA', 0.8, 15.0, 0.7, 2, 0.2),  # High relationship, high dependency
            ('AMD', 0.6, 8.0, 0.4, 3, 0.1),
            ('Intel', 0.7, 12.0, 0.5, 3, 0.1),
            ('Broadcom', 0.5, 5.0, 0.3, 4, 0.2),
            ('Marvell', 0.4, 2.0, 0.2, 5, 0.2),
        ]
        
        for name, strength, value, dependency, alternatives, geo_risk in suppliers:
            self.supplier_relationships[name] = SupplierRelationship(
                supplier_name=name,
                agent_id=None,  # Will be linked during simulation setup
                relationship_strength=strength,
                contract_value_billions=value * (self.annual_capex / 50),  # Scale by CAPEX
                dependency_score=dependency,
                alternative_sources=alternatives,
                geopolitical_risk=geo_risk
            )
    
    def step(self):
        """Execute one simulation step."""
        # Update market conditions awareness
        self._assess_market_conditions()
        
        # Make strategic decisions
        decisions = self.make_strategic_decisions()
        
        # Execute capacity planning
        self._execute_capacity_planning()
        
        # Manage supplier relationships
        self._manage_supplier_relationships()
        
        # Update financial metrics
        self._update_financial_performance()
        
        # Track performance
        self.update_metrics()
        
        return decisions
    
    def make_strategic_decisions(self) -> Dict[str, Any]:
        """Make strategic decisions for this time step."""
        decisions = {}
        
        # 1. Data center expansion decisions
        decisions['dc_expansion'] = self._decide_datacenter_expansion()
        
        # 2. Chip procurement strategy
        decisions['procurement'] = self._decide_procurement_strategy()
        
        # 3. Technology investment allocation
        decisions['tech_investment'] = self._decide_technology_investments()
        
        # 4. Geographic diversification
        decisions['geographic'] = self._decide_geographic_strategy()
        
        # 5. Supplier diversification
        decisions['supplier_diversification'] = self._decide_supplier_diversification()
        
        return decisions
    
    def _decide_datacenter_expansion(self) -> Dict[str, Any]:
        """Decide on data center expansion strategy."""
        # Calculate demand growth forecast
        ai_demand_growth = 0.3 + self.random.normal(0, 0.05)  # 30% ± 5%
        market_share_pressure = max(0, 0.15 - self.cloud_market_share)  # Pressure if < 15%
        
        # Assess constraints
        capital_availability = min(1.0, self.capital / (self.annual_capex * 0.5))
        regulatory_ease = 1.0 - self.assess_geopolitical_risk()
        
        # Decision logic
        expansion_pressure = (
            ai_demand_growth * 0.4 +
            market_share_pressure * 0.3 +
            capital_availability * 0.2 +
            self.expansion_aggressiveness * 0.1
        )
        
        if expansion_pressure > 0.7:
            strategy = DataCenterStrategy.AGGRESSIVE_GROWTH
            new_capacity_mw = self.annual_capex * 0.6 / 2.5  # $2.5B per 1000MW
        elif expansion_pressure > 0.4:
            strategy = DataCenterStrategy.MODERATE_EXPANSION
            new_capacity_mw = self.annual_capex * 0.4 / 2.5
        elif expansion_pressure > 0.2:
            strategy = DataCenterStrategy.GEOGRAPHIC_DIVERSIFICATION
            new_capacity_mw = self.annual_capex * 0.2 / 2.5
        else:
            strategy = DataCenterStrategy.CONSOLIDATION
            new_capacity_mw = 0
        
        self.current_dc_strategy = strategy
        
        return {
            'strategy': strategy.value,
            'new_capacity_mw': new_capacity_mw,
            'expansion_pressure': expansion_pressure,
            'capex_allocation': new_capacity_mw * 2.5
        }
    
    def _decide_procurement_strategy(self) -> Dict[str, Any]:
        """Decide chip procurement strategy based on risk assessment."""
        # Assess supply chain risks
        geopolitical_risk = self.assess_geopolitical_risk()
        supplier_concentration = self._calculate_supplier_concentration()
        demand_volatility = 0.2 + self.random.normal(0, 0.05)
        
        # Calculate strategy score
        diversification_pressure = (
            geopolitical_risk * 0.4 +
            supplier_concentration * 0.3 +
            demand_volatility * 0.2 +
            (1 - self.risk_tolerance) * 0.1
        )
        
        if diversification_pressure > 0.8:
            strategy = ChipProcurementStrategy.MULTI_SOURCE
            target_suppliers = 4
        elif diversification_pressure > 0.6:
            strategy = ChipProcurementStrategy.DUAL_SOURCE
            target_suppliers = 2
        elif diversification_pressure > 0.3:
            strategy = ChipProcurementStrategy.SINGLE_SOURCE
            target_suppliers = 1
        else:
            strategy = ChipProcurementStrategy.VERTICAL_INTEGRATION
            target_suppliers = 0  # Internal development
        
        self.current_procurement_strategy = strategy
        
        return {
            'strategy': strategy.value,
            'target_suppliers': target_suppliers,
            'diversification_pressure': diversification_pressure,
            'geopolitical_risk': geopolitical_risk
        }
    
    def _decide_technology_investments(self) -> Dict[str, float]:
        """Allocate R&D budget across technology areas."""
        rd_budget = self.annual_capex * 0.15  # 15% of CAPEX for R&D
        
        # Base allocation weights
        weights = {
            'quantum_computing': 0.2,
            'neuromorphic_chips': 0.15,
            'optical_computing': 0.1,
            'edge_infrastructure': 0.55
        }
        
        # Adjust based on strategic priorities
        if self.current_dc_strategy == DataCenterStrategy.AGGRESSIVE_GROWTH:
            weights['edge_infrastructure'] += 0.1
            weights['quantum_computing'] -= 0.05
            weights['neuromorphic_chips'] -= 0.05
        
        # Geopolitical risk affects quantum/advanced tech investment
        geo_risk = self.assess_geopolitical_risk()
        if geo_risk > 0.5:
            weights['quantum_computing'] += 0.1 * geo_risk
            weights['edge_infrastructure'] -= 0.1 * geo_risk
        
        # Allocate budget
        allocations = {}
        for tech, weight in weights.items():
            allocations[tech] = rd_budget * weight
            self.technology_investments[tech] += allocations[tech]
        
        return allocations
    
    def _decide_geographic_strategy(self) -> Dict[str, Any]:
        """Decide geographic expansion priorities."""
        # Assess regional attractiveness
        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']
        regional_scores = {}
        
        for region in regions:
            # Scoring factors
            market_potential = self.random.uniform(0.3, 0.9)
            regulatory_ease = self.random.uniform(0.2, 0.8)
            cost_advantage = self.random.uniform(0.1, 0.9)
            geo_stability = self.random.uniform(0.4, 0.9)
            
            # Adjust for known factors
            if region == 'Asia Pacific':
                geo_stability *= 0.7  # Higher geopolitical risk
                cost_advantage *= 1.2  # Lower costs
            elif region == 'Europe':
                regulatory_ease *= 0.8  # More regulation
                market_potential *= 1.1  # Good demand
            
            regional_scores[region] = (
                market_potential * 0.3 +
                regulatory_ease * 0.25 +
                cost_advantage * 0.25 +
                geo_stability * 0.2
            )
        
        # Select top regions for expansion
        sorted_regions = sorted(regional_scores.items(), key=lambda x: x[1], reverse=True)
        priority_regions = sorted_regions[:3]
        
        return {
            'priority_regions': [r[0] for r in priority_regions],
            'regional_scores': regional_scores,
            'expansion_budget': self.annual_capex * 0.3
        }
    
    def _decide_supplier_diversification(self) -> Dict[str, Any]:
        """Decide on supplier portfolio management."""
        actions = []
        
        for supplier_name, relationship in self.supplier_relationships.items():
            # Assess relationship health
            risk_score = (
                relationship.dependency_score * 0.4 +
                relationship.geopolitical_risk * 0.3 +
                (1 / max(1, relationship.alternative_sources)) * 0.3
            )
            
            if risk_score > 0.7:
                actions.append({
                    'action': 'diversify',
                    'supplier': supplier_name,
                    'risk_score': risk_score,
                    'target_dependency_reduction': 0.2
                })
            elif risk_score < 0.3 and relationship.relationship_strength > 0.8:
                actions.append({
                    'action': 'deepen',
                    'supplier': supplier_name,
                    'risk_score': risk_score,
                    'contract_increase': 0.1
                })
        
        return {
            'actions': actions,
            'total_suppliers': len(self.supplier_relationships),
            'avg_dependency': np.mean([r.dependency_score for r in self.supplier_relationships.values()])
        }
    
    def _calculate_supplier_concentration(self) -> float:
        """Calculate supplier concentration using Herfindahl index."""
        if not self.supplier_relationships:
            return 1.0
        
        total_value = sum(r.contract_value_billions for r in self.supplier_relationships.values())
        if total_value == 0:
            return 1.0
        
        shares = [r.contract_value_billions / total_value for r in self.supplier_relationships.values()]
        hhi = sum(share ** 2 for share in shares)
        return hhi
    
    def _assess_market_conditions(self):
        """Assess current market conditions and update internal state."""
        # This would integrate with market dynamics models in Phase 3
        # For now, use simplified assessment
        
        # AI demand growth
        self.ai_workload_percentage = min(0.8, self.ai_workload_percentage + self.random.normal(0.02, 0.01))
        
        # Competition pressure
        total_market_share = getattr(self.model, 'total_hyperscaler_market_share', 1.0)
        competitive_pressure = max(0, total_market_share - 1.0)
        
        # Update capacity utilization
        demand_growth = 0.025 + self.random.normal(0, 0.01)  # 2.5% monthly growth baseline
        self.capacity_utilization = min(0.95, self.capacity_utilization + demand_growth)
    
    def _execute_capacity_planning(self):
        """Execute capacity planning decisions."""
        # Implement data center expansion plans
        completed_plans = []
        for plan in self.dc_expansion_plans:
            plan.timeline_months -= 1
            if plan.timeline_months <= 0:
                # Plan completed
                self.data_center_count += 1
                self.capital -= plan.estimated_cost_billions
                completed_plans.append(plan)
        
        # Remove completed plans
        for plan in completed_plans:
            self.dc_expansion_plans.remove(plan)
    
    def _manage_supplier_relationships(self):
        """Manage and update supplier relationships."""
        for supplier_name, relationship in self.supplier_relationships.items():
            # Relationship strength evolves based on performance and market conditions
            performance_factor = self.random.normal(0, 0.02)  # Random performance variation
            geo_risk_impact = -relationship.geopolitical_risk * 0.01  # Geo risk degrades relationships
            
            relationship.relationship_strength = np.clip(
                relationship.relationship_strength + performance_factor + geo_risk_impact,
                0.0, 1.0
            )
    
    def _update_financial_performance(self):
        """Update financial performance metrics."""
        # Revenue growth from capacity expansion
        revenue_growth = self.capacity_utilization * 0.02 + self.cloud_market_share * 0.01
        
        # Costs from expansion and procurement
        expansion_costs = sum(plan.estimated_cost_billions for plan in self.dc_expansion_plans) * 0.1
        procurement_costs = sum(r.contract_value_billions for r in self.supplier_relationships.values()) * 0.05
        
        # Update financials
        monthly_revenue = self.annual_capex * 0.8 * (1 + revenue_growth)  # Revenue roughly 80% of CAPEX
        monthly_costs = expansion_costs + procurement_costs + monthly_revenue * 0.7  # 70% cost ratio
        
        self.revenue_history.append(monthly_revenue)
        self.cost_history.append(monthly_costs)
        
        # Update capital
        monthly_profit = monthly_revenue - monthly_costs
        self.capital += monthly_profit
    
    def assess_geopolitical_risk(self) -> float:
        """Assess current geopolitical risk level."""
        # Base risk from global tensions
        base_risk = 0.3  # 30% baseline geopolitical risk
        
        # Add supplier-specific risks
        supplier_risk = np.mean([r.geopolitical_risk for r in self.supplier_relationships.values()])
        
        # Add concentration risk
        concentration_risk = self._calculate_supplier_concentration() * 0.2
        
        # Add geographic exposure risk
        geographic_risk = 0.1  # Simplified for now
        
        total_risk = base_risk + supplier_risk * 0.3 + concentration_risk + geographic_risk
        return np.clip(total_risk, 0.0, 1.0)
    
    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get summary of current strategic position."""
        return {
            'agent_id': self.unique_id,
            'name': self.name,
            'cloud_market_share': self.cloud_market_share,
            'annual_capex': self.annual_capex,
            'data_centers': self.data_center_count,
            'capacity_utilization': self.capacity_utilization,
            'current_strategies': {
                'datacenter': self.current_dc_strategy.value,
                'procurement': self.current_procurement_strategy.value
            },
            'risk_metrics': {
                'geopolitical_risk': self.assess_geopolitical_risk(),
                'supplier_concentration': self._calculate_supplier_concentration(),
                'capacity_utilization': self.capacity_utilization
            },
            'supplier_count': len(self.supplier_relationships),
            'expansion_plans': len(self.dc_expansion_plans),
            'technology_investments': dict(self.technology_investments)
        } 