"""
Critical Path Analysis for Semiconductor Supply Chain

Identifies critical bottlenecks, dependencies, and vulnerability points in the
semiconductor supply chain network using advanced graph analysis techniques.
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from enum import Enum
import heapq

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class NodeType(Enum):
    """Supply chain node types."""
    RAW_MATERIALS = "raw_materials"
    EQUIPMENT_SUPPLIER = "equipment_supplier"
    COMPONENT_SUPPLIER = "component_supplier"
    FOUNDRY = "foundry"
    PACKAGING_ASSEMBLY = "packaging_assembly"
    TESTING = "testing"
    DISTRIBUTION = "distribution"
    OEM = "oem"

class CriticalityLevel(Enum):
    """Criticality classification levels."""
    CRITICAL = "critical"        # Single point of failure
    HIGH = "high"               # Limited alternatives
    MEDIUM = "medium"           # Some alternatives available
    LOW = "low"                 # Multiple alternatives

@dataclass
class SupplyChainNode:
    """Supply chain network node."""
    node_id: str
    node_type: NodeType
    name: str
    location: str
    capacity: float
    reliability: float  # 0-1 reliability score
    lead_time_days: int
    strategic_importance: float  # 0-1 importance score
    alternative_suppliers: int
    geopolitical_risk: float  # 0-1 risk score
    dependencies: List[str]  # Upstream dependencies
    customers: List[str]     # Downstream customers

@dataclass
class CriticalPath:
    """Critical path through supply chain."""
    path_id: str
    nodes: List[str]
    total_lead_time: int
    bottleneck_capacity: float
    risk_score: float
    criticality_level: CriticalityLevel
    vulnerability_points: List[str]
    mitigation_strategies: List[str]

@dataclass
class BottleneckAnalysis:
    """Bottleneck identification and analysis."""
    bottleneck_node: str
    capacity_constraint: float
    demand_pressure: float
    expansion_cost: float
    expansion_time_months: int
    alternative_routes: List[List[str]]
    impact_if_disrupted: float

class CriticalPathAnalyzer:
    """
    Advanced critical path analysis for semiconductor supply chains.
    
    Performs:
    - Critical path identification using graph algorithms
    - Bottleneck detection and capacity analysis
    - Single point of failure identification
    - Risk propagation analysis
    - Alternative route evaluation
    - Resilience optimization recommendations
    """
    
    def __init__(self):
        # Supply chain network graph
        self.supply_network = nx.DiGraph()
        
        # Node and edge data
        self.nodes: Dict[str, SupplyChainNode] = {}
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        
        # Analysis results
        self.critical_paths: List[CriticalPath] = []
        self.bottlenecks: List[BottleneckAnalysis] = []
        self.vulnerability_scores: Dict[str, float] = {}
        
        # Initialize with realistic supply chain structure
        self._initialize_supply_chain_network()
    
    def _initialize_supply_chain_network(self):
        """Initialize realistic semiconductor supply chain network."""
        
        # Define supply chain nodes with realistic data
        supply_chain_data = [
            # Raw Materials
            ("silicon_wafers", NodeType.RAW_MATERIALS, "Silicon Wafer Production", "Japan", 1000000, 0.95, 30, 0.9, 3, 0.2),
            ("photoresist", NodeType.RAW_MATERIALS, "Photoresist Production", "Japan", 50000, 0.90, 45, 0.8, 2, 0.3),
            ("rare_earth_materials", NodeType.RAW_MATERIALS, "Rare Earth Materials", "China", 100000, 0.85, 60, 0.95, 1, 0.8),
            ("ultra_pure_chemicals", NodeType.RAW_MATERIALS, "Ultra-pure Chemicals", "Europe", 75000, 0.92, 21, 0.7, 4, 0.2),
            
            # Equipment Suppliers
            ("asml_euv", NodeType.EQUIPMENT_SUPPLIER, "ASML EUV Systems", "Netherlands", 50, 0.85, 365, 0.98, 0, 0.3),
            ("applied_materials", NodeType.EQUIPMENT_SUPPLIER, "Applied Materials Equipment", "USA", 500, 0.88, 180, 0.85, 2, 0.2),
            ("tokyo_electron", NodeType.EQUIPMENT_SUPPLIER, "Tokyo Electron", "Japan", 300, 0.90, 120, 0.80, 3, 0.25),
            ("lam_research", NodeType.EQUIPMENT_SUPPLIER, "Lam Research Etch", "USA", 400, 0.87, 150, 0.75, 3, 0.2),
            
            # Component Suppliers
            ("advanced_packaging", NodeType.COMPONENT_SUPPLIER, "Advanced Packaging", "Taiwan", 10000, 0.88, 45, 0.85, 5, 0.4),
            ("substrates", NodeType.COMPONENT_SUPPLIER, "IC Substrates", "Japan", 50000, 0.92, 30, 0.80, 4, 0.25),
            ("testing_equipment", NodeType.COMPONENT_SUPPLIER, "Testing Equipment", "USA", 200, 0.90, 90, 0.70, 3, 0.2),
            
            # Foundries
            ("tsmc_advanced", NodeType.FOUNDRY, "TSMC Advanced Nodes", "Taiwan", 150000, 0.92, 90, 0.95, 2, 0.6),
            ("samsung_foundry", NodeType.FOUNDRY, "Samsung Foundry", "South_Korea", 120000, 0.88, 75, 0.85, 2, 0.4),
            ("intel_foundry", NodeType.FOUNDRY, "Intel Foundry", "USA", 100000, 0.85, 60, 0.80, 1, 0.2),
            ("globalfoundries", NodeType.FOUNDRY, "GlobalFoundries", "USA", 80000, 0.83, 45, 0.70, 3, 0.2),
            ("smic", NodeType.FOUNDRY, "SMIC", "China", 90000, 0.80, 90, 0.60, 1, 0.8),
            
            # Assembly & Test
            ("ase_taiwan", NodeType.PACKAGING_ASSEMBLY, "ASE Taiwan", "Taiwan", 500000, 0.90, 21, 0.85, 4, 0.4),
            ("amkor_korea", NodeType.PACKAGING_ASSEMBLY, "Amkor Korea", "South_Korea", 400000, 0.88, 18, 0.80, 4, 0.3),
            ("jcet_china", NodeType.PACKAGING_ASSEMBLY, "JCET China", "China", 600000, 0.85, 15, 0.70, 3, 0.7),
            
            # Testing
            ("advantest", NodeType.TESTING, "Advantest Testing", "Japan", 100000, 0.92, 30, 0.85, 2, 0.25),
            ("teradyne", NodeType.TESTING, "Teradyne Testing", "USA", 80000, 0.90, 35, 0.80, 3, 0.2),
            
            # OEMs (Major customers)
            ("apple", NodeType.OEM, "Apple", "Global", 1000000, 0.95, 7, 0.90, 5, 0.3),
            ("nvidia", NodeType.OEM, "NVIDIA", "USA", 500000, 0.93, 14, 0.95, 3, 0.3),
            ("amd", NodeType.OEM, "AMD", "USA", 300000, 0.90, 21, 0.85, 4, 0.2),
            ("qualcomm", NodeType.OEM, "Qualcomm", "USA", 400000, 0.91, 28, 0.88, 3, 0.3),
        ]
        
        # Create nodes
        for node_id, node_type, name, location, capacity, reliability, lead_time, importance, alternatives, geo_risk in supply_chain_data:
            node = SupplyChainNode(
                node_id=node_id,
                node_type=node_type,
                name=name,
                location=location,
                capacity=capacity,
                reliability=reliability,
                lead_time_days=lead_time,
                strategic_importance=importance,
                alternative_suppliers=alternatives,
                geopolitical_risk=geo_risk,
                dependencies=[],
                customers=[]
            )
            self.nodes[node_id] = node
            self.supply_network.add_node(node_id, **node.__dict__)
        
        # Define supply chain relationships (edges)
        supply_relationships = [
            # Raw materials to equipment/components
            ("silicon_wafers", "tsmc_advanced", 0.95),
            ("silicon_wafers", "samsung_foundry", 0.90),
            ("silicon_wafers", "intel_foundry", 0.85),
            ("silicon_wafers", "globalfoundries", 0.80),
            ("silicon_wafers", "smic", 0.75),
            
            ("photoresist", "tsmc_advanced", 0.90),
            ("photoresist", "samsung_foundry", 0.85),
            ("photoresist", "intel_foundry", 0.80),
            
            ("rare_earth_materials", "asml_euv", 0.95),
            ("rare_earth_materials", "applied_materials", 0.80),
            ("rare_earth_materials", "tokyo_electron", 0.75),
            
            ("ultra_pure_chemicals", "tsmc_advanced", 0.85),
            ("ultra_pure_chemicals", "samsung_foundry", 0.82),
            ("ultra_pure_chemicals", "intel_foundry", 0.80),
            
            # Equipment to foundries
            ("asml_euv", "tsmc_advanced", 0.98),
            ("asml_euv", "samsung_foundry", 0.95),
            ("asml_euv", "intel_foundry", 0.90),
            
            ("applied_materials", "tsmc_advanced", 0.90),
            ("applied_materials", "samsung_foundry", 0.88),
            ("applied_materials", "intel_foundry", 0.92),
            ("applied_materials", "globalfoundries", 0.85),
            ("applied_materials", "smic", 0.70),
            
            ("tokyo_electron", "tsmc_advanced", 0.85),
            ("tokyo_electron", "samsung_foundry", 0.90),
            ("tokyo_electron", "smic", 0.75),
            
            ("lam_research", "tsmc_advanced", 0.88),
            ("lam_research", "samsung_foundry", 0.85),
            ("lam_research", "intel_foundry", 0.90),
            ("lam_research", "globalfoundries", 0.82),
            
            # Components to assembly
            ("substrates", "ase_taiwan", 0.90),
            ("substrates", "amkor_korea", 0.85),
            ("substrates", "jcet_china", 0.80),
            
            ("advanced_packaging", "ase_taiwan", 0.95),
            ("advanced_packaging", "amkor_korea", 0.90),
            
            # Foundries to assembly
            ("tsmc_advanced", "ase_taiwan", 0.95),
            ("tsmc_advanced", "amkor_korea", 0.85),
            ("samsung_foundry", "amkor_korea", 0.95),
            ("samsung_foundry", "ase_taiwan", 0.80),
            ("intel_foundry", "ase_taiwan", 0.70),
            ("globalfoundries", "ase_taiwan", 0.75),
            ("globalfoundries", "amkor_korea", 0.80),
            ("smic", "jcet_china", 0.95),
            
            # Assembly to testing
            ("ase_taiwan", "advantest", 0.90),
            ("ase_taiwan", "teradyne", 0.85),
            ("amkor_korea", "advantest", 0.88),
            ("amkor_korea", "teradyne", 0.90),
            ("jcet_china", "advantest", 0.75),
            ("jcet_china", "teradyne", 0.70),
            
            # Testing to OEMs
            ("advantest", "apple", 0.95),
            ("advantest", "nvidia", 0.90),
            ("advantest", "amd", 0.85),
            ("advantest", "qualcomm", 0.88),
            
            ("teradyne", "apple", 0.90),
            ("teradyne", "nvidia", 0.95),
            ("teradyne", "amd", 0.90),
            ("teradyne", "qualcomm", 0.85),
        ]
        
        # Add edges with weights
        for source, target, reliability in supply_relationships:
            if source in self.nodes and target in self.nodes:
                # Edge weight combines reliability, lead time, and capacity factors
                source_node = self.nodes[source]
                target_node = self.nodes[target]
                
                # Calculate composite weight (lower is better for shortest path)
                lead_time_factor = (source_node.lead_time_days + target_node.lead_time_days) / 100
                reliability_factor = 2 - reliability  # Invert reliability (lower is better)
                capacity_factor = 1 / (min(source_node.capacity, target_node.capacity) / 10000)
                
                weight = lead_time_factor + reliability_factor + capacity_factor
                
                self.supply_network.add_edge(source, target, weight=weight, reliability=reliability)
                self.edge_weights[(source, target)] = weight
                
                # Update node relationships
                self.nodes[source].customers.append(target)
                self.nodes[target].dependencies.append(source)
    
    def find_critical_paths(self, source_nodes: Optional[List[str]] = None, 
                          target_nodes: Optional[List[str]] = None) -> List[CriticalPath]:
        """Find critical paths through the supply chain network."""
        
        if source_nodes is None:
            source_nodes = [node_id for node_id, node in self.nodes.items() 
                          if node.node_type == NodeType.RAW_MATERIALS]
        
        if target_nodes is None:
            target_nodes = [node_id for node_id, node in self.nodes.items() 
                          if node.node_type == NodeType.OEM]
        
        critical_paths = []
        
        # Find shortest paths from each source to each target
        for source in source_nodes:
            for target in target_nodes:
                if source in self.supply_network and target in self.supply_network:
                    try:
                        # Find shortest path by weight (considers lead time, reliability, capacity)
                        path = nx.shortest_path(self.supply_network, source, target, weight='weight')
                        
                        if len(path) > 1:  # Valid path found
                            critical_path = self._analyze_path(path, f"{source}_to_{target}")
                            critical_paths.append(critical_path)
                    
                    except nx.NetworkXNoPath:
                        # No path exists between source and target
                        continue
        
        # Sort by criticality and risk
        critical_paths.sort(key=lambda p: (p.criticality_level.value, -p.risk_score))
        
        self.critical_paths = critical_paths
        return critical_paths
    
    def _analyze_path(self, path: List[str], path_id: str) -> CriticalPath:
        """Analyze a specific path for criticality and risk."""
        
        # Calculate total lead time
        total_lead_time = sum(self.nodes[node_id].lead_time_days for node_id in path)
        
        # Calculate bottleneck capacity (minimum along path)
        bottleneck_capacity = min(self.nodes[node_id].capacity for node_id in path)
        
        # Calculate path risk score
        path_risks = []
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            
            # Node risks
            source_risk = (1 - self.nodes[source].reliability) + self.nodes[source].geopolitical_risk
            target_risk = (1 - self.nodes[target].reliability) + self.nodes[target].geopolitical_risk
            
            # Edge risk (lower reliability = higher risk)
            edge_reliability = self.supply_network[source][target].get('reliability', 0.8)
            edge_risk = 1 - edge_reliability
            
            path_risks.append((source_risk + target_risk + edge_risk) / 3)
        
        risk_score = np.mean(path_risks) if path_risks else 0
        
        # Identify vulnerability points (nodes with high risk or low alternatives)
        vulnerability_points = []
        for node_id in path:
            node = self.nodes[node_id]
            vulnerability = (node.geopolitical_risk + (1 - node.reliability) + 
                           (1 / max(1, node.alternative_suppliers)))
            
            if vulnerability > 1.5:  # Threshold for high vulnerability
                vulnerability_points.append(node_id)
        
        # Determine criticality level
        single_points_of_failure = [node_id for node_id in path 
                                  if self.nodes[node_id].alternative_suppliers == 0]
        
        if single_points_of_failure:
            criticality_level = CriticalityLevel.CRITICAL
        elif risk_score > 0.7:
            criticality_level = CriticalityLevel.HIGH
        elif risk_score > 0.4:
            criticality_level = CriticalityLevel.MEDIUM
        else:
            criticality_level = CriticalityLevel.LOW
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(path, vulnerability_points)
        
        return CriticalPath(
            path_id=path_id,
            nodes=path,
            total_lead_time=total_lead_time,
            bottleneck_capacity=bottleneck_capacity,
            risk_score=risk_score,
            criticality_level=criticality_level,
            vulnerability_points=vulnerability_points,
            mitigation_strategies=mitigation_strategies
        )
    
    def _generate_mitigation_strategies(self, path: List[str], vulnerability_points: List[str]) -> List[str]:
        """Generate mitigation strategies for critical path vulnerabilities."""
        strategies = []
        
        # Generic strategies based on path characteristics
        if vulnerability_points:
            strategies.append("Diversify supplier base for vulnerable nodes")
            strategies.append("Develop alternative supply routes")
            strategies.append("Increase strategic inventory for critical components")
        
        # Node-specific strategies
        for node_id in vulnerability_points:
            node = self.nodes[node_id]
            
            if node.geopolitical_risk > 0.5:
                strategies.append(f"Establish geographically diversified alternatives to {node.name}")
            
            if node.alternative_suppliers < 2:
                strategies.append(f"Qualify additional suppliers for {node.name}")
            
            if node.reliability < 0.85:
                strategies.append(f"Improve reliability of {node.name} through quality programs")
        
        # Path-level strategies
        if len(path) > 6:  # Long path
            strategies.append("Evaluate vertical integration opportunities")
        
        if any(self.nodes[node_id].node_type == NodeType.FOUNDRY for node_id in vulnerability_points):
            strategies.append("Consider foundry partnerships or capacity investments")
        
        return list(set(strategies))  # Remove duplicates
    
    def identify_bottlenecks(self, demand_forecast: Optional[Dict[str, float]] = None) -> List[BottleneckAnalysis]:
        """Identify capacity bottlenecks in the supply chain."""
        
        if demand_forecast is None:
            # Use default demand assumptions
            demand_forecast = {node_id: node.capacity * 0.8 for node_id, node in self.nodes.items()}
        
        bottlenecks = []
        
        for node_id, node in self.nodes.items():
            demand = demand_forecast.get(node_id, 0)
            capacity = node.capacity
            
            # Calculate capacity utilization
            utilization = demand / capacity if capacity > 0 else 0
            
            # Identify bottlenecks (high utilization or capacity constraints)
            if utilization > 0.9 or capacity < demand:
                # Calculate demand pressure
                demand_pressure = max(0, (demand - capacity) / capacity) if capacity > 0 else 1.0
                
                # Estimate expansion parameters
                expansion_cost = self._estimate_expansion_cost(node)
                expansion_time = self._estimate_expansion_time(node)
                
                # Find alternative routes
                alternative_routes = self._find_alternative_routes(node_id)
                
                # Calculate impact if disrupted
                impact = self._calculate_disruption_impact(node_id)
                
                bottleneck = BottleneckAnalysis(
                    bottleneck_node=node_id,
                    capacity_constraint=capacity,
                    demand_pressure=demand_pressure,
                    expansion_cost=expansion_cost,
                    expansion_time_months=expansion_time,
                    alternative_routes=alternative_routes,
                    impact_if_disrupted=impact
                )
                
                bottlenecks.append(bottleneck)
        
        # Sort by impact and demand pressure
        bottlenecks.sort(key=lambda b: (b.impact_if_disrupted, b.demand_pressure), reverse=True)
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def _estimate_expansion_cost(self, node: SupplyChainNode) -> float:
        """Estimate cost to expand node capacity."""
        # Cost factors by node type (billions USD)
        cost_factors = {
            NodeType.RAW_MATERIALS: 0.5,
            NodeType.EQUIPMENT_SUPPLIER: 2.0,
            NodeType.COMPONENT_SUPPLIER: 1.0,
            NodeType.FOUNDRY: 20.0,  # Very expensive fab construction
            NodeType.PACKAGING_ASSEMBLY: 0.5,
            NodeType.TESTING: 0.3,
            NodeType.DISTRIBUTION: 0.1,
            NodeType.OEM: 1.0
        }
        
        base_cost = cost_factors.get(node.node_type, 1.0)
        
        # Adjust for location (some regions more expensive)
        location_multipliers = {
            "USA": 1.3,
            "Europe": 1.2,
            "Japan": 1.1,
            "Taiwan": 1.0,
            "South_Korea": 0.9,
            "China": 0.7
        }
        
        location_multiplier = location_multipliers.get(node.location, 1.0)
        
        # Adjust for current capacity (larger expansions more expensive)
        capacity_factor = (node.capacity / 100000) ** 0.5
        
        return base_cost * location_multiplier * capacity_factor
    
    def _estimate_expansion_time(self, node: SupplyChainNode) -> int:
        """Estimate time to expand node capacity (months)."""
        # Time factors by node type
        time_factors = {
            NodeType.RAW_MATERIALS: 12,
            NodeType.EQUIPMENT_SUPPLIER: 24,
            NodeType.COMPONENT_SUPPLIER: 18,
            NodeType.FOUNDRY: 48,  # Long fab construction time
            NodeType.PACKAGING_ASSEMBLY: 12,
            NodeType.TESTING: 9,
            NodeType.DISTRIBUTION: 6,
            NodeType.OEM: 24
        }
        
        base_time = time_factors.get(node.node_type, 18)
        
        # Adjust for geopolitical complexity
        geo_adjustment = 1 + node.geopolitical_risk * 0.5
        
        return int(base_time * geo_adjustment)
    
    def _find_alternative_routes(self, node_id: str) -> List[List[str]]:
        """Find alternative routes that bypass a specific node."""
        alternative_routes = []
        
        # Find paths that don't include this node
        try:
            # Get predecessors and successors
            predecessors = list(self.supply_network.predecessors(node_id))
            successors = list(self.supply_network.successors(node_id))
            
            # Try to find paths from predecessors to successors that bypass this node
            temp_network = self.supply_network.copy()
            temp_network.remove_node(node_id)
            
            for pred in predecessors:
                for succ in successors:
                    try:
                        alt_path = nx.shortest_path(temp_network, pred, succ)
                        if len(alt_path) > 1:
                            alternative_routes.append(alt_path)
                    except nx.NetworkXNoPath:
                        continue
            
        except:
            pass
        
        return alternative_routes[:3]  # Return top 3 alternatives
    
    def _calculate_disruption_impact(self, node_id: str) -> float:
        """Calculate the impact if a node is disrupted."""
        node = self.nodes[node_id]
        
        # Base impact from strategic importance
        base_impact = node.strategic_importance
        
        # Multiply by lack of alternatives
        alternative_factor = 1 / max(1, node.alternative_suppliers)
        
        # Consider downstream dependencies
        downstream_count = len(list(self.supply_network.successors(node_id)))
        downstream_factor = 1 + (downstream_count * 0.1)
        
        # Consider capacity importance
        capacity_factor = min(2.0, node.capacity / 50000)  # Normalize capacity impact
        
        impact = base_impact * alternative_factor * downstream_factor * capacity_factor
        
        return min(1.0, impact)  # Cap at 1.0
    
    def calculate_vulnerability_scores(self) -> Dict[str, float]:
        """Calculate vulnerability scores for all nodes."""
        vulnerability_scores = {}
        
        for node_id, node in self.nodes.items():
            # Multiple vulnerability factors
            reliability_risk = 1 - node.reliability
            geopolitical_risk = node.geopolitical_risk
            alternative_risk = 1 / max(1, node.alternative_suppliers)
            capacity_risk = self._calculate_capacity_risk(node_id)
            
            # Weighted combination
            vulnerability = (
                reliability_risk * 0.25 +
                geopolitical_risk * 0.30 +
                alternative_risk * 0.25 +
                capacity_risk * 0.20
            )
            
            vulnerability_scores[node_id] = min(1.0, vulnerability)
        
        self.vulnerability_scores = vulnerability_scores
        return vulnerability_scores
    
    def _calculate_capacity_risk(self, node_id: str) -> float:
        """Calculate capacity-related risk for a node."""
        node = self.nodes[node_id]
        
        # Risk increases with longer lead times
        lead_time_risk = min(1.0, node.lead_time_days / 365)
        
        # Risk from being a capacity bottleneck
        downstream_demand = len(list(self.supply_network.successors(node_id)))
        capacity_pressure = min(1.0, downstream_demand / 10)
        
        return (lead_time_risk + capacity_pressure) / 2
    
    def generate_resilience_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations to improve supply chain resilience."""
        recommendations = []
        
        # Analyze critical paths for recommendations
        if not self.critical_paths:
            self.find_critical_paths()
        
        # Analyze bottlenecks
        if not self.bottlenecks:
            self.identify_bottlenecks()
        
        # Vulnerability analysis
        if not self.vulnerability_scores:
            self.calculate_vulnerability_scores()
        
        # Generate recommendations based on analysis
        
        # 1. Address critical single points of failure
        critical_nodes = [node_id for node_id, score in self.vulnerability_scores.items() if score > 0.8]
        for node_id in critical_nodes[:5]:  # Top 5 most vulnerable
            node = self.nodes[node_id]
            recommendations.append({
                'type': 'critical_vulnerability',
                'priority': 'high',
                'node': node_id,
                'description': f"Address critical vulnerability in {node.name}",
                'specific_actions': [
                    f"Qualify {max(2, 3 - node.alternative_suppliers)} additional suppliers",
                    f"Establish strategic inventory buffer of {int(node.capacity * 0.1)} units",
                    f"Improve reliability from {node.reliability:.2f} to {min(0.95, node.reliability + 0.1):.2f}"
                ],
                'estimated_cost_millions': self._estimate_expansion_cost(node) * 1000 * 0.3,
                'implementation_time_months': 12,
                'risk_reduction': 0.3
            })
        
        # 2. Address capacity bottlenecks
        for bottleneck in self.bottlenecks[:3]:  # Top 3 bottlenecks
            node = self.nodes[bottleneck.bottleneck_node]
            recommendations.append({
                'type': 'capacity_expansion',
                'priority': 'high' if bottleneck.demand_pressure > 0.5 else 'medium',
                'node': bottleneck.bottleneck_node,
                'description': f"Expand capacity at {node.name}",
                'specific_actions': [
                    f"Increase capacity by {int(node.capacity * 0.3)} units",
                    f"Reduce lead time from {node.lead_time_days} to {max(7, int(node.lead_time_days * 0.8))} days",
                    "Implement demand forecasting and capacity planning systems"
                ],
                'estimated_cost_millions': bottleneck.expansion_cost * 1000,
                'implementation_time_months': bottleneck.expansion_time_months,
                'risk_reduction': 0.4
            })
        
        # 3. Geographic diversification
        geo_concentrations = self._analyze_geographic_concentration()
        for region, concentration in geo_concentrations.items():
            if concentration > 0.6:  # High concentration risk
                recommendations.append({
                    'type': 'geographic_diversification',
                    'priority': 'medium',
                    'region': region,
                    'description': f"Reduce over-concentration in {region}",
                    'specific_actions': [
                        f"Establish suppliers in alternative regions",
                        f"Reduce {region} dependency from {concentration:.1%} to <50%",
                        "Implement region-specific risk monitoring"
                    ],
                    'estimated_cost_millions': 500,
                    'implementation_time_months': 24,
                    'risk_reduction': 0.25
                })
        
        # 4. Technology and process improvements
        tech_nodes = [node_id for node_id, node in self.nodes.items() 
                     if node.node_type in [NodeType.EQUIPMENT_SUPPLIER, NodeType.FOUNDRY]]
        for node_id in tech_nodes:
            node = self.nodes[node_id]
            if node.reliability < 0.9:
                recommendations.append({
                    'type': 'technology_improvement',
                    'priority': 'medium',
                    'node': node_id,
                    'description': f"Improve technology and processes at {node.name}",
                    'specific_actions': [
                        "Implement predictive maintenance systems",
                        "Upgrade equipment and automation",
                        "Enhance quality control processes"
                    ],
                    'estimated_cost_millions': 100,
                    'implementation_time_months': 18,
                    'risk_reduction': 0.15
                })
        
        # Sort recommendations by priority and potential impact
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda r: (priority_order[r['priority']], r['risk_reduction']), reverse=True)
        
        return recommendations
    
    def _analyze_geographic_concentration(self) -> Dict[str, float]:
        """Analyze geographic concentration of supply chain nodes."""
        region_counts = {}
        total_nodes = len(self.nodes)
        
        for node in self.nodes.values():
            region = node.location
            region_counts[region] = region_counts.get(region, 0) + 1
        
        return {region: count / total_nodes for region, count in region_counts.items()}
    
    def get_critical_path_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of critical path analysis."""
        if not self.critical_paths:
            self.find_critical_paths()
        if not self.bottlenecks:
            self.identify_bottlenecks()
        if not self.vulnerability_scores:
            self.calculate_vulnerability_scores()
        
        # Critical path statistics
        criticality_distribution = {}
        for path in self.critical_paths:
            level = path.criticality_level.value
            criticality_distribution[level] = criticality_distribution.get(level, 0) + 1
        
        # Vulnerability statistics
        high_vulnerability_nodes = sum(1 for score in self.vulnerability_scores.values() if score > 0.7)
        avg_vulnerability = np.mean(list(self.vulnerability_scores.values()))
        
        # Bottleneck statistics
        critical_bottlenecks = sum(1 for b in self.bottlenecks if b.demand_pressure > 0.5)
        
        return {
            'total_supply_chain_nodes': len(self.nodes),
            'total_critical_paths': len(self.critical_paths),
            'criticality_distribution': criticality_distribution,
            'high_vulnerability_nodes': high_vulnerability_nodes,
            'average_vulnerability_score': avg_vulnerability,
            'total_bottlenecks': len(self.bottlenecks),
            'critical_bottlenecks': critical_bottlenecks,
            'geographic_concentration': self._analyze_geographic_concentration(),
            'most_vulnerable_nodes': sorted(self.vulnerability_scores.items(), 
                                          key=lambda x: x[1], reverse=True)[:5],
            'longest_critical_path': max(p.total_lead_time for p in self.critical_paths) if self.critical_paths else 0,
            'highest_risk_path': max(p.risk_score for p in self.critical_paths) if self.critical_paths else 0
        } 