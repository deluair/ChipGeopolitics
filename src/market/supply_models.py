"""
Supply Capacity Models

Implements sophisticated supply modeling for semiconductor manufacturing including
capacity constraints, fab utilization, technology node capabilities, and expansion planning.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class CapacityType(Enum):
    """Manufacturing capacity types."""
    WAFER_CAPACITY = "wafer_capacity"  # Monthly wafer capacity
    PACKAGING_CAPACITY = "packaging_capacity"  # Die packaging capacity
    TESTING_CAPACITY = "testing_capacity"  # Chip testing capacity
    ASSEMBLY_CAPACITY = "assembly_capacity"  # Final assembly

class ConstraintType(Enum):
    """Supply constraint types."""
    EQUIPMENT_BOTTLENECK = "equipment_bottleneck"
    MATERIAL_SHORTAGE = "material_shortage"
    SKILLED_LABOR = "skilled_labor"
    UTILITY_INFRASTRUCTURE = "utility_infrastructure"
    GEOPOLITICAL_RESTRICTION = "geopolitical_restriction"
    TECHNOLOGY_MATURITY = "technology_maturity"

@dataclass
class FabCapacityData:
    """Fab capacity and constraint information."""
    fab_id: str
    location: str
    owner: str
    process_nodes: List[str]  # Supported process nodes
    monthly_wafer_capacity: Dict[str, int]  # Capacity by node
    utilization_rate: float  # Current utilization
    equipment_vintage: Dict[str, float]  # Equipment age by process step
    yield_rates: Dict[str, float]  # Yield by process node
    expansion_timeline: Dict[str, int]  # Months to complete expansions
    constraint_factors: Dict[ConstraintType, float]  # Constraint severity (0-1)

@dataclass
class CapacityConstraints:
    """Supply chain capacity constraints."""
    constraint_type: ConstraintType
    affected_process_nodes: List[str]
    affected_regions: List[str]
    severity_level: float  # 0-1, where 1 is most severe
    duration_months: int
    mitigation_cost_billions: float
    alternative_sources: int
    strategic_importance: float

@dataclass
class SupplyChainNode:
    """Supply chain network node."""
    node_id: str
    node_type: str  # 'fab', 'packaging', 'testing', 'materials'
    location: str
    capacity_data: Dict[str, float]
    dependencies: List[str]  # Upstream dependencies
    customers: List[str]  # Downstream customers
    risk_factors: Dict[str, float]

class SupplyCapacityModel:
    """
    Comprehensive supply capacity modeling for semiconductor manufacturing.
    
    Models:
    - Fab capacity and utilization dynamics
    - Supply chain bottlenecks and constraints
    - Technology node transition timelines
    - Geographic distribution of capacity
    - Equipment and material dependencies
    - Expansion planning and lead times
    """
    
    def __init__(self):
        # Global capacity tracking
        self.fab_capacity_data: Dict[str, FabCapacityData] = {}
        self.supply_chain_network: Dict[str, SupplyChainNode] = {}
        self.active_constraints: List[CapacityConstraints] = []
        
        # Market dynamics
        self.demand_capacity_ratio = {}  # By process node
        self.capacity_utilization_targets = {}
        self.expansion_pipeline = {}
        
        # Initialize with realistic industry data
        self._initialize_global_capacity()
        self._initialize_supply_chain_network()
        self._initialize_baseline_constraints()
    
    def _initialize_global_capacity(self):
        """Initialize global semiconductor manufacturing capacity."""
        # Major fabs with realistic capacity data
        fab_configs = [
            # TSMC fabs
            ("TSMC_Fab18", "Taiwan", "TSMC", ["3nm", "5nm"], {"3nm": 15000, "5nm": 50000}, 0.95, 2.0),
            ("TSMC_Fab15", "Taiwan", "TSMC", ["5nm", "7nm"], {"5nm": 40000, "7nm": 80000}, 0.92, 3.0),
            ("TSMC_Fab14", "Taiwan", "TSMC", ["7nm", "10nm"], {"7nm": 60000, "10nm": 100000}, 0.90, 4.0),
            ("TSMC_Arizona", "USA", "TSMC", ["5nm"], {"5nm": 20000}, 0.0, 1.0),  # Under construction
            
            # Samsung fabs
            ("Samsung_S3", "South_Korea", "Samsung", ["3nm", "5nm"], {"3nm": 10000, "5nm": 30000}, 0.88, 2.5),
            ("Samsung_S2", "South_Korea", "Samsung", ["7nm", "10nm"], {"7nm": 40000, "10nm": 60000}, 0.85, 5.0),
            ("Samsung_Texas", "USA", "Samsung", ["5nm"], {"5nm": 15000}, 0.0, 1.0),  # Under construction
            
            # Intel fabs
            ("Intel_Fab42", "USA", "Intel", ["Intel7", "Intel4"], {"Intel7": 50000, "Intel4": 20000}, 0.80, 4.0),
            ("Intel_Fab32", "USA", "Intel", ["Intel7", "10nm"], {"Intel7": 40000, "10nm": 30000}, 0.82, 6.0),
            ("Intel_Ireland", "Europe", "Intel", ["Intel7"], {"Intel7": 25000}, 0.78, 5.0),
            
            # GlobalFoundries
            ("GF_Malta", "USA", "GlobalFoundries", ["12nm", "14nm"], {"12nm": 30000, "14nm": 50000}, 0.75, 6.0),
            ("GF_Dresden", "Europe", "GlobalFoundries", ["12nm", "22nm"], {"12nm": 25000, "22nm": 40000}, 0.78, 7.0),
            
            # SMIC (China)
            ("SMIC_Shanghai", "China", "SMIC", ["7nm", "14nm"], {"7nm": 15000, "14nm": 80000}, 0.85, 4.0),
            ("SMIC_Beijing", "China", "SMIC", ["14nm", "28nm"], {"14nm": 40000, "28nm": 120000}, 0.88, 5.0),
            
            # Other Asian fabs
            ("UMC_Taiwan", "Taiwan", "UMC", ["14nm", "28nm"], {"14nm": 35000, "28nm": 100000}, 0.82, 6.0),
            ("Tower_Israel", "Middle_East", "Tower", ["65nm", "180nm"], {"65nm": 20000, "180nm": 50000}, 0.80, 8.0)
        ]
        
        # Initialize fab capacity data
        for fab_id, location, owner, nodes, capacity, utilization, vintage in fab_configs:
            # Generate realistic yield rates
            yield_rates = {}
            for node in nodes:
                if "3nm" in node:
                    yield_rates[node] = 0.65
                elif "5nm" in node:
                    yield_rates[node] = 0.75
                elif "7nm" in node:
                    yield_rates[node] = 0.85
                else:
                    yield_rates[node] = 0.90
            
            # Generate equipment vintage by process step
            equipment_vintage = {
                'lithography': vintage + np.random.uniform(-1, 1),
                'etch': vintage + np.random.uniform(-1, 1),
                'deposition': vintage + np.random.uniform(-0.5, 1.5),
                'metrology': vintage + np.random.uniform(-0.5, 0.5),
                'cmp': vintage + np.random.uniform(-1, 2)
            }
            
            # Generate constraint factors
            constraint_factors = {
                ConstraintType.EQUIPMENT_BOTTLENECK: np.random.uniform(0.1, 0.4),
                ConstraintType.MATERIAL_SHORTAGE: np.random.uniform(0.05, 0.3),
                ConstraintType.SKILLED_LABOR: np.random.uniform(0.1, 0.5),
                ConstraintType.UTILITY_INFRASTRUCTURE: np.random.uniform(0.05, 0.2),
                ConstraintType.GEOPOLITICAL_RESTRICTION: 0.1 if "China" in location else 0.05,
                ConstraintType.TECHNOLOGY_MATURITY: np.random.uniform(0.1, 0.3)
            }
            
            # Expansion timeline (simplified)
            expansion_timeline = {"capacity_expansion": np.random.randint(12, 36)}
            
            self.fab_capacity_data[fab_id] = FabCapacityData(
                fab_id=fab_id,
                location=location,
                owner=owner,
                process_nodes=nodes,
                monthly_wafer_capacity=capacity,
                utilization_rate=utilization,
                equipment_vintage=equipment_vintage,
                yield_rates=yield_rates,
                expansion_timeline=expansion_timeline,
                constraint_factors=constraint_factors
            )
    
    def _initialize_supply_chain_network(self):
        """Initialize supply chain network nodes."""
        # Critical supply chain nodes
        supply_nodes = [
            # Equipment suppliers
            ("ASML_Netherlands", "equipment", "Europe", {"EUV_systems": 50, "DUV_systems": 200}, [], []),
            ("Applied_Materials_US", "equipment", "USA", {"etch_systems": 500, "deposition": 300}, [], []),
            ("Tokyo_Electron_Japan", "equipment", "Japan", {"coater_developer": 200, "etch": 150}, [], []),
            
            # Materials suppliers
            ("Shin_Etsu_Japan", "materials", "Japan", {"silicon_wafers": 5000000, "photoresist": 1000}, [], []),
            ("SUMCO_Japan", "materials", "Japan", {"silicon_wafers": 4000000}, [], []),
            ("JSR_Japan", "materials", "Japan", {"photoresist": 800, "CMP_slurry": 500}, [], []),
            
            # Packaging and testing
            ("ASE_Taiwan", "packaging", "Taiwan", {"packaging_capacity": 10000000}, [], []),
            ("Amkor_Korea", "packaging", "South_Korea", {"packaging_capacity": 8000000}, [], []),
            ("JCET_China", "packaging", "China", {"packaging_capacity": 12000000}, [], []),
        ]
        
        for node_id, node_type, location, capacity, deps, customers in supply_nodes:
            # Generate risk factors based on location and type
            risk_factors = {
                'geopolitical_risk': 0.3 if "China" in location else 0.1,
                'natural_disaster_risk': 0.4 if location in ["Japan", "Taiwan"] else 0.2,
                'supply_concentration_risk': 0.6 if node_type == "equipment" else 0.3,
                'technology_obsolescence_risk': 0.2
            }
            
            self.supply_chain_network[node_id] = SupplyChainNode(
                node_id=node_id,
                node_type=node_type,
                location=location,
                capacity_data=capacity,
                dependencies=deps,
                customers=customers,
                risk_factors=risk_factors
            )
    
    def _initialize_baseline_constraints(self):
        """Initialize baseline supply chain constraints."""
        baseline_constraints = [
            # EUV equipment bottleneck
            CapacityConstraints(
                constraint_type=ConstraintType.EQUIPMENT_BOTTLENECK,
                affected_process_nodes=["3nm", "5nm"],
                affected_regions=["Global"],
                severity_level=0.7,
                duration_months=24,
                mitigation_cost_billions=5.0,
                alternative_sources=1,  # Only ASML
                strategic_importance=0.95
            ),
            
            # Advanced materials shortage
            CapacityConstraints(
                constraint_type=ConstraintType.MATERIAL_SHORTAGE,
                affected_process_nodes=["3nm", "5nm", "7nm"],
                affected_regions=["Global"],
                severity_level=0.4,
                duration_months=12,
                mitigation_cost_billions=2.0,
                alternative_sources=3,
                strategic_importance=0.8
            ),
            
            # Skilled labor shortage
            CapacityConstraints(
                constraint_type=ConstraintType.SKILLED_LABOR,
                affected_process_nodes=["All"],
                affected_regions=["USA", "Europe"],
                severity_level=0.5,
                duration_months=36,
                mitigation_cost_billions=10.0,
                alternative_sources=2,  # Training, immigration
                strategic_importance=0.7
            ),
            
            # Geopolitical restrictions
            CapacityConstraints(
                constraint_type=ConstraintType.GEOPOLITICAL_RESTRICTION,
                affected_process_nodes=["7nm", "5nm", "3nm"],
                affected_regions=["China"],
                severity_level=0.8,
                duration_months=60,
                mitigation_cost_billions=50.0,
                alternative_sources=0,
                strategic_importance=0.9
            )
        ]
        
        self.active_constraints.extend(baseline_constraints)
    
    def calculate_effective_capacity(self, process_node: str, region: Optional[str] = None) -> Dict[str, float]:
        """Calculate effective manufacturing capacity considering constraints."""
        effective_capacity = {}
        
        # Filter fabs by criteria
        relevant_fabs = []
        for fab_id, fab_data in self.fab_capacity_data.items():
            if process_node in fab_data.process_nodes:
                if region is None or fab_data.location == region:
                    relevant_fabs.append((fab_id, fab_data))
        
        # Calculate effective capacity for each fab
        for fab_id, fab_data in relevant_fabs:
            base_capacity = fab_data.monthly_wafer_capacity.get(process_node, 0)
            
            # Apply utilization
            utilized_capacity = base_capacity * fab_data.utilization_rate
            
            # Apply yield rates
            yield_rate = fab_data.yield_rates.get(process_node, 0.8)
            yielded_capacity = utilized_capacity * yield_rate
            
            # Apply constraint impacts
            constraint_impact = self._calculate_constraint_impact(fab_data, process_node)
            constrained_capacity = yielded_capacity * (1 - constraint_impact)
            
            effective_capacity[fab_id] = {
                'base_capacity': base_capacity,
                'utilized_capacity': utilized_capacity,
                'yielded_capacity': yielded_capacity,
                'effective_capacity': constrained_capacity,
                'constraint_impact': constraint_impact,
                'owner': fab_data.owner,
                'location': fab_data.location
            }
        
        return effective_capacity
    
    def _calculate_constraint_impact(self, fab_data: FabCapacityData, process_node: str) -> float:
        """Calculate total constraint impact on fab capacity."""
        total_impact = 0.0
        
        for constraint in self.active_constraints:
            # Check if constraint affects this fab/node
            if (process_node in constraint.affected_process_nodes or 
                "All" in constraint.affected_process_nodes):
                
                if (fab_data.location in constraint.affected_regions or
                    "Global" in constraint.affected_regions):
                    
                    # Calculate constraint impact
                    base_impact = constraint.severity_level
                    
                    # Adjust for fab-specific factors
                    if constraint.constraint_type in fab_data.constraint_factors:
                        fab_factor = fab_data.constraint_factors[constraint.constraint_type]
                        adjusted_impact = base_impact * fab_factor
                    else:
                        adjusted_impact = base_impact * 0.5  # Default factor
                    
                    total_impact += adjusted_impact
        
        # Cap total impact at 0.8 (80% maximum constraint)
        return min(0.8, total_impact)
    
    def calculate_supply_demand_balance(self, demand_forecast: Dict[str, List[float]], 
                                      time_horizon_quarters: int) -> Dict[str, Any]:
        """Calculate supply-demand balance across process nodes."""
        balance_analysis = {}
        
        # Get current effective capacity by node
        process_nodes = ["3nm", "5nm", "7nm", "10nm", "16nm", "28nm"]
        
        for node in process_nodes:
            effective_capacity = self.calculate_effective_capacity(node)
            total_monthly_capacity = sum(fab['effective_capacity'] for fab in effective_capacity.values())
            quarterly_capacity = total_monthly_capacity * 3  # Convert to quarterly
            
            # Get demand forecast for this node
            node_demand = demand_forecast.get(node, [0] * time_horizon_quarters)
            
            # Calculate balance metrics
            supply_demand_ratios = []
            capacity_utilization = []
            shortage_quarters = 0
            
            for quarter in range(min(len(node_demand), time_horizon_quarters)):
                demand = node_demand[quarter]
                ratio = quarterly_capacity / demand if demand > 0 else float('inf')
                utilization = demand / quarterly_capacity if quarterly_capacity > 0 else 1.0
                
                supply_demand_ratios.append(ratio)
                capacity_utilization.append(utilization)
                
                if ratio < 1.0:
                    shortage_quarters += 1
            
            # Calculate stress metrics
            avg_utilization = np.mean(capacity_utilization) if capacity_utilization else 0
            peak_utilization = max(capacity_utilization) if capacity_utilization else 0
            shortage_probability = shortage_quarters / len(capacity_utilization) if capacity_utilization else 0
            
            balance_analysis[node] = {
                'total_quarterly_capacity': quarterly_capacity,
                'avg_utilization': avg_utilization,
                'peak_utilization': peak_utilization,
                'shortage_probability': shortage_probability,
                'supply_demand_ratios': supply_demand_ratios,
                'fab_count': len(effective_capacity),
                'geographic_distribution': self._calculate_geographic_distribution(effective_capacity),
                'owner_concentration': self._calculate_owner_concentration(effective_capacity)
            }
        
        return balance_analysis
    
    def _calculate_geographic_distribution(self, effective_capacity: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate geographic distribution of capacity."""
        regional_capacity = {}
        total_capacity = sum(fab['effective_capacity'] for fab in effective_capacity.values())
        
        for fab_data in effective_capacity.values():
            location = fab_data['location']
            capacity = fab_data['effective_capacity']
            
            if location not in regional_capacity:
                regional_capacity[location] = 0
            regional_capacity[location] += capacity
        
        # Convert to percentages
        if total_capacity > 0:
            for region in regional_capacity:
                regional_capacity[region] /= total_capacity
        
        return regional_capacity
    
    def _calculate_owner_concentration(self, effective_capacity: Dict[str, Dict]) -> float:
        """Calculate ownership concentration using Herfindahl index."""
        owner_capacity = {}
        total_capacity = sum(fab['effective_capacity'] for fab in effective_capacity.values())
        
        for fab_data in effective_capacity.values():
            owner = fab_data['owner']
            capacity = fab_data['effective_capacity']
            
            if owner not in owner_capacity:
                owner_capacity[owner] = 0
            owner_capacity[owner] += capacity
        
        # Calculate HHI
        if total_capacity > 0:
            shares = [capacity / total_capacity for capacity in owner_capacity.values()]
            hhi = sum(share ** 2 for share in shares)
            return hhi
        
        return 1.0
    
    def simulate_capacity_expansion(self, expansion_plans: Dict[str, Dict[str, float]], 
                                  timeline_quarters: int) -> Dict[str, List[float]]:
        """Simulate capacity expansion over time."""
        # Track capacity evolution by process node
        node_capacity_evolution = {}
        process_nodes = ["3nm", "5nm", "7nm", "10nm", "16nm", "28nm"]
        
        for node in process_nodes:
            quarterly_capacity = []
            
            for quarter in range(timeline_quarters):
                # Start with current capacity
                current_capacity = self.calculate_effective_capacity(node)
                total_capacity = sum(fab['effective_capacity'] for fab in current_capacity.values())
                
                # Add expansions that come online this quarter
                for fab_id, expansion_data in expansion_plans.items():
                    if fab_id in self.fab_capacity_data:
                        fab_data = self.fab_capacity_data[fab_id]
                        if node in fab_data.process_nodes:
                            # Check if expansion comes online this quarter
                            expansion_quarter = expansion_data.get('start_quarter', 0)
                            expansion_duration = expansion_data.get('duration_quarters', 12)
                            
                            if expansion_quarter <= quarter < expansion_quarter + expansion_duration:
                                # Linear ramp-up of capacity
                                progress = (quarter - expansion_quarter) / expansion_duration
                                additional_capacity = expansion_data.get('additional_capacity', 0) * progress
                                total_capacity += additional_capacity
                
                quarterly_capacity.append(total_capacity)
            
            node_capacity_evolution[node] = quarterly_capacity
        
        return node_capacity_evolution
    
    def assess_supply_chain_risks(self) -> Dict[str, Any]:
        """Assess supply chain vulnerability and risk factors."""
        risk_assessment = {
            'overall_risk_score': 0.0,
            'critical_dependencies': [],
            'geographic_risks': {},
            'technology_risks': {},
            'mitigation_strategies': []
        }
        
        # Assess geographic concentration risks
        geographic_risks = {}
        for region in ["Taiwan", "South_Korea", "China", "USA", "Europe"]:
            region_capacity = 0
            total_capacity = 0
            
            for fab_data in self.fab_capacity_data.values():
                if fab_data.location == region:
                    fab_capacity = sum(fab_data.monthly_wafer_capacity.values())
                    region_capacity += fab_capacity
                total_capacity += sum(fab_data.monthly_wafer_capacity.values())
            
            concentration = region_capacity / total_capacity if total_capacity > 0 else 0
            
            # Risk scoring based on concentration and regional risk factors
            base_risk = concentration * 2  # Concentration penalty
            if region == "Taiwan":
                base_risk *= 1.5  # Geopolitical risk multiplier
            elif region == "China":
                base_risk *= 1.3  # Trade restrictions
            
            geographic_risks[region] = {
                'capacity_share': concentration,
                'risk_score': min(1.0, base_risk),
                'risk_factors': ['geopolitical_tension', 'natural_disasters'] if region in ["Taiwan", "China"] else ['regulatory_changes']
            }
        
        risk_assessment['geographic_risks'] = geographic_risks
        
        # Assess technology node risks
        technology_risks = {}
        for node in ["3nm", "5nm", "7nm"]:
            capacity_data = self.calculate_effective_capacity(node)
            
            # Calculate concentration by owner
            owner_capacity = {}
            total_node_capacity = sum(fab['effective_capacity'] for fab in capacity_data.values())
            
            for fab_data in capacity_data.values():
                owner = fab_data['owner']
                if owner not in owner_capacity:
                    owner_capacity[owner] = 0
                owner_capacity[owner] += fab_data['effective_capacity']
            
            # Calculate HHI for concentration risk
            if total_node_capacity > 0:
                shares = [cap / total_node_capacity for cap in owner_capacity.values()]
                concentration_risk = sum(share ** 2 for share in shares)
            else:
                concentration_risk = 1.0
            
            technology_risks[node] = {
                'concentration_risk': concentration_risk,
                'total_capacity': total_node_capacity,
                'primary_suppliers': list(owner_capacity.keys()),
                'supply_security': 1 - concentration_risk
            }
        
        risk_assessment['technology_risks'] = technology_risks
        
        # Identify critical dependencies
        critical_deps = []
        for constraint in self.active_constraints:
            if constraint.strategic_importance > 0.8 and constraint.alternative_sources < 2:
                critical_deps.append({
                    'constraint_type': constraint.constraint_type.value,
                    'affected_nodes': constraint.affected_process_nodes,
                    'severity': constraint.severity_level,
                    'alternatives': constraint.alternative_sources
                })
        
        risk_assessment['critical_dependencies'] = critical_deps
        
        # Calculate overall risk score
        geo_risk = max(risk['risk_score'] for risk in geographic_risks.values())
        tech_risk = max(risk['concentration_risk'] for risk in technology_risks.values())
        constraint_risk = np.mean([c.severity_level for c in self.active_constraints])
        
        risk_assessment['overall_risk_score'] = (geo_risk * 0.4 + tech_risk * 0.3 + constraint_risk * 0.3)
        
        return risk_assessment
    
    def add_supply_constraint(self, constraint: CapacityConstraints):
        """Add a new supply chain constraint."""
        self.active_constraints.append(constraint)
    
    def remove_supply_constraint(self, constraint_type: ConstraintType, affected_regions: List[str]):
        """Remove supply constraints matching criteria."""
        self.active_constraints = [
            c for c in self.active_constraints 
            if not (c.constraint_type == constraint_type and 
                   any(region in c.affected_regions for region in affected_regions))
        ]
    
    def update_fab_utilization(self, fab_id: str, new_utilization: float):
        """Update fab utilization rate."""
        if fab_id in self.fab_capacity_data:
            self.fab_capacity_data[fab_id].utilization_rate = np.clip(new_utilization, 0.0, 1.0)
    
    def get_supply_summary(self) -> Dict[str, Any]:
        """Get comprehensive supply capacity summary."""
        # Calculate total global capacity
        total_capacity_by_node = {}
        total_fabs_by_region = {}
        
        for fab_data in self.fab_capacity_data.values():
            # Count fabs by region
            region = fab_data.location
            if region not in total_fabs_by_region:
                total_fabs_by_region[region] = 0
            total_fabs_by_region[region] += 1
            
            # Sum capacity by node
            for node, capacity in fab_data.monthly_wafer_capacity.items():
                if node not in total_capacity_by_node:
                    total_capacity_by_node[node] = 0
                total_capacity_by_node[node] += capacity
        
        # Get supply chain risk assessment
        risk_assessment = self.assess_supply_chain_risks()
        
        return {
            'total_fabs': len(self.fab_capacity_data),
            'fabs_by_region': total_fabs_by_region,
            'capacity_by_node_monthly': total_capacity_by_node,
            'active_constraints': len(self.active_constraints),
            'supply_chain_nodes': len(self.supply_chain_network),
            'overall_risk_score': risk_assessment['overall_risk_score'],
            'critical_dependencies': len(risk_assessment['critical_dependencies']),
            'geographic_concentration': max(risk_assessment['geographic_risks'][region]['capacity_share'] 
                                          for region in risk_assessment['geographic_risks']),
            'technology_concentration': max(risk_assessment['technology_risks'][node]['concentration_risk']
                                          for node in risk_assessment['technology_risks']),
            'average_utilization': np.mean([fab.utilization_rate for fab in self.fab_capacity_data.values()])
        } 