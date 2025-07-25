"""
Network Resilience Analysis for Semiconductor Supply Chains

Provides comprehensive resilience metrics, redundancy analysis, and stress testing
for semiconductor supply chain networks using advanced graph theory and network analysis.
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict, deque

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class ResilienceMetric(Enum):
    """Types of resilience metrics."""
    STRUCTURAL_RESILIENCE = "structural_resilience"
    FUNCTIONAL_RESILIENCE = "functional_resilience"
    ADAPTIVE_RESILIENCE = "adaptive_resilience"
    ROBUSTNESS = "robustness"
    REDUNDANCY = "redundancy"
    DIVERSITY = "diversity"
    EFFICIENCY = "efficiency"
    FLEXIBILITY = "flexibility"

class StressTestType(Enum):
    """Types of stress tests."""
    NODE_REMOVAL = "node_removal"
    EDGE_REMOVAL = "edge_removal"
    CAPACITY_REDUCTION = "capacity_reduction"
    DEMAND_SURGE = "demand_surge"
    GEOGRAPHIC_DISRUPTION = "geographic_disruption"
    SUPPLIER_CONCENTRATION = "supplier_concentration"
    CASCADE_FAILURE = "cascade_failure"

@dataclass
class ResilienceScore:
    """Resilience scoring container."""
    metric_type: ResilienceMetric
    score: float  # 0-1 scale
    components: Dict[str, float]
    confidence_interval: Tuple[float, float]
    interpretation: str

@dataclass
class RedundancyAnalysis:
    """Redundancy analysis results."""
    node_id: str
    primary_paths: int
    backup_paths: int
    path_diversity: float
    geographic_redundancy: float
    supplier_redundancy: float
    technology_redundancy: float
    overall_redundancy: float

@dataclass
class StressTestResult:
    """Stress test result container."""
    test_type: StressTestType
    test_parameters: Dict[str, Any]
    resilience_degradation: float  # 0-1 scale
    affected_nodes: List[str]
    critical_failures: List[str]
    recovery_time_estimate: int  # Days
    mitigation_strategies: List[str]

class NetworkResilienceAnalyzer:
    """
    Advanced network resilience analysis for semiconductor supply chains.
    
    Provides:
    - Comprehensive resilience metrics
    - Redundancy and diversity analysis
    - Network stress testing
    - Recovery time estimation
    - Vulnerability identification
    - Resilience optimization recommendations
    """
    
    def __init__(self, supply_network: nx.DiGraph):
        self.supply_network = supply_network.copy()
        
        # Analysis results
        self.resilience_scores: Dict[ResilienceMetric, ResilienceScore] = {}
        self.redundancy_analysis: Dict[str, RedundancyAnalysis] = {}
        self.stress_test_results: List[StressTestResult] = []
        
        # Network properties
        self.network_metrics: Dict[str, float] = {}
        self.centrality_measures: Dict[str, Dict[str, float]] = {}
        self.community_structure: Dict[str, List[str]] = {}
        
        # Initialize analysis
        self._calculate_network_metrics()
        self._calculate_centrality_measures()
        self._detect_community_structure()
    
    def _calculate_network_metrics(self):
        """Calculate basic network topology metrics."""
        G = self.supply_network
        
        # Basic metrics
        self.network_metrics.update({
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_degree': sum(degree for _, degree in G.degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
            'clustering_coefficient': nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 0 else 0,
            'transitivity': nx.transitivity(G.to_undirected())
        })
        
        # Path-based metrics
        try:
            if nx.is_strongly_connected(G) and G.number_of_nodes() > 1:
                self.network_metrics.update({
                    'average_path_length': nx.average_shortest_path_length(G),
                    'diameter': nx.diameter(G),
                    'radius': nx.radius(G)
                })
        except:
            # Default values for disconnected or trivial graphs
            self.network_metrics.update({
                'average_path_length': float('inf'),
                'diameter': float('inf'), 
                'radius': float('inf')
            })
        
        # Connectivity metrics
        try:
            undirected_G = G.to_undirected()
            self.network_metrics.update({
                'connected_components': nx.number_connected_components(undirected_G),
                'node_connectivity': nx.node_connectivity(undirected_G),
                'edge_connectivity': nx.edge_connectivity(undirected_G)
            })
        except:
            self.network_metrics.update({
                'connected_components': 1,
                'node_connectivity': 0,
                'edge_connectivity': 0
            })
    
    def _calculate_centrality_measures(self):
        """Calculate centrality measures for all nodes."""
        G = self.supply_network
        
        # Initialize with zeros
        nodes = list(G.nodes())
        default_centrality = {node: 0.0 for node in nodes}
        
        # Various centrality measures
        centrality_functions = {
            'degree_centrality': nx.degree_centrality,
            'betweenness_centrality': nx.betweenness_centrality,
            'closeness_centrality': nx.closeness_centrality,
            'pagerank': nx.pagerank
        }
        
        for measure_name, centrality_func in centrality_functions.items():
            try:
                if measure_name == 'closeness_centrality':
                    # Use undirected graph for closeness centrality
                    centrality = centrality_func(G.to_undirected())
                else:
                    centrality = centrality_func(G)
                
                self.centrality_measures[measure_name] = centrality
            except:
                # Fallback for problematic networks
                self.centrality_measures[measure_name] = default_centrality.copy()
    
    def _detect_community_structure(self):
        """Detect community structure in the network."""
        try:
            # Convert to undirected for community detection
            undirected_G = self.supply_network.to_undirected()
            
            # Simple community detection using connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(undirected_G)):
                communities[f"component_{i}"] = list(component)
            
            self.community_structure = communities
            
        except Exception:
            # Fallback: single community
            self.community_structure = {"component_0": list(self.supply_network.nodes())}
    
    def calculate_structural_resilience(self) -> ResilienceScore:
        """Calculate structural resilience based on network topology."""
        G = self.supply_network
        
        # Component scores
        components = {}
        
        # Network density (0-1, higher is more resilient)
        components['density'] = self.network_metrics.get('density', 0)
        
        # Clustering coefficient (0-1, higher indicates better local resilience)
        components['clustering'] = self.network_metrics.get('clustering_coefficient', 0)
        
        # Connectivity (normalized by network size)
        node_conn = self.network_metrics.get('node_connectivity', 0)
        edge_conn = self.network_metrics.get('edge_connectivity', 0)
        max_possible_conn = min(G.number_of_nodes() - 1, G.number_of_edges()) if G.number_of_nodes() > 1 else 0
        
        if max_possible_conn > 0:
            components['node_connectivity'] = node_conn / max_possible_conn
            components['edge_connectivity'] = edge_conn / max_possible_conn
        else:
            components['node_connectivity'] = 0
            components['edge_connectivity'] = 0
        
        # Path redundancy (inverse of average path length, normalized)
        avg_path = self.network_metrics.get('average_path_length', float('inf'))
        if avg_path != float('inf') and avg_path > 0:
            components['path_efficiency'] = 1.0 / avg_path
        else:
            components['path_efficiency'] = 0
        
        # Calculate overall score (weighted average)
        weights = {
            'density': 0.2,
            'clustering': 0.2,
            'node_connectivity': 0.25,
            'edge_connectivity': 0.2,
            'path_efficiency': 0.15
        }
        
        overall_score = sum(weights[k] * components[k] for k in weights.keys())
        
        # Confidence interval (simplified)
        confidence_interval = (
            max(0, overall_score - 0.1),
            min(1, overall_score + 0.1)
        )
        
        # Interpretation
        if overall_score >= 0.8:
            interpretation = "Highly resilient network structure"
        elif overall_score >= 0.6:
            interpretation = "Moderately resilient network structure"
        elif overall_score >= 0.4:
            interpretation = "Limited network resilience"
        else:
            interpretation = "Low network resilience, vulnerable to disruptions"
        
        score = ResilienceScore(
            metric_type=ResilienceMetric.STRUCTURAL_RESILIENCE,
            score=overall_score,
            components=components,
            confidence_interval=confidence_interval,
            interpretation=interpretation
        )
        
        self.resilience_scores[ResilienceMetric.STRUCTURAL_RESILIENCE] = score
        return score
    
    def calculate_functional_resilience(self) -> ResilienceScore:
        """Calculate functional resilience based on capacity and flow."""
        G = self.supply_network
        
        # Component scores
        components = {}
        
        # Capacity utilization analysis
        total_capacity = 0
        total_flow = 0
        capacity_variance = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            capacity = node_data.get('capacity', 100)
            current_flow = node_data.get('current_flow', capacity * 0.7)  # Assume 70% utilization
            
            total_capacity += capacity
            total_flow += current_flow
            
            if capacity > 0:
                utilization = current_flow / capacity
                capacity_variance.append(utilization)
        
        # Capacity headroom (unused capacity as resilience buffer)
        if total_capacity > 0:
            components['capacity_headroom'] = (total_capacity - total_flow) / total_capacity
        else:
            components['capacity_headroom'] = 0
        
        # Capacity distribution (lower variance = better resilience)
        if capacity_variance:
            components['capacity_balance'] = 1.0 - np.std(capacity_variance)
        else:
            components['capacity_balance'] = 0
        
        # Flow diversity (multiple paths for critical flows)
        critical_nodes = [n for n in G.nodes() if self.centrality_measures.get('betweenness_centrality', {}).get(n, 0) > 0.1]
        if critical_nodes:
            # Simplified flow diversity calculation
            components['flow_diversity'] = min(1.0, len(critical_nodes) / max(1, G.number_of_nodes() * 0.3))
        else:
            components['flow_diversity'] = 0.5  # Default moderate diversity
        
        # Bottleneck analysis
        bottleneck_score = 1.0
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            if in_degree == 1 or out_degree == 1:  # Potential bottleneck
                bottleneck_score *= 0.9
        
        components['bottleneck_resilience'] = bottleneck_score
        
        # Calculate overall score
        weights = {
            'capacity_headroom': 0.3,
            'capacity_balance': 0.25,
            'flow_diversity': 0.25,
            'bottleneck_resilience': 0.2
        }
        
        overall_score = sum(weights[k] * components[k] for k in weights.keys())
        
        # Confidence interval
        confidence_interval = (
            max(0, overall_score - 0.15),
            min(1, overall_score + 0.15)
        )
        
        # Interpretation
        if overall_score >= 0.8:
            interpretation = "High functional resilience with good capacity buffers"
        elif overall_score >= 0.6:
            interpretation = "Adequate functional resilience"
        elif overall_score >= 0.4:
            interpretation = "Limited functional resilience, monitor capacity"
        else:
            interpretation = "Low functional resilience, capacity constraints likely"
        
        score = ResilienceScore(
            metric_type=ResilienceMetric.FUNCTIONAL_RESILIENCE,
            score=overall_score,
            components=components,
            confidence_interval=confidence_interval,
            interpretation=interpretation
        )
        
        self.resilience_scores[ResilienceMetric.FUNCTIONAL_RESILIENCE] = score
        return score
    
    def analyze_redundancy(self) -> Dict[str, RedundancyAnalysis]:
        """Analyze redundancy for each node in the network."""
        G = self.supply_network
        redundancy_results = {}
        
        for node in G.nodes():
            node_data = G.nodes[node]
            
            # Path analysis
            primary_paths = 0
            backup_paths = 0
            
            # Count paths to/from this node
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))
            
            primary_paths = len(predecessors) + len(successors)
            
            # Backup paths (indirect connections)
            backup_paths = 0
            for pred in predecessors:
                backup_paths += len(list(G.predecessors(pred)))
            for succ in successors:
                backup_paths += len(list(G.successors(succ)))
            
            # Path diversity (based on different routes)
            total_paths = primary_paths + backup_paths
            path_diversity = min(1.0, total_paths / 10.0) if total_paths > 0 else 0
            
            # Geographic redundancy (simplified)
            geographic_redundancy = 0.7  # Default moderate geographic spread
            
            # Supplier redundancy
            supplier_count = len(predecessors)
            supplier_redundancy = min(1.0, supplier_count / 5.0)  # Normalize by ideal count
            
            # Technology redundancy (simplified)
            technology_redundancy = 0.6  # Default moderate tech diversity
            
            # Overall redundancy score
            redundancy_factors = [
                path_diversity,
                geographic_redundancy,
                supplier_redundancy,
                technology_redundancy
            ]
            overall_redundancy = float(np.mean(redundancy_factors))
            
            analysis = RedundancyAnalysis(
                node_id=node,
                primary_paths=primary_paths,
                backup_paths=backup_paths,
                path_diversity=path_diversity,
                geographic_redundancy=geographic_redundancy,
                supplier_redundancy=supplier_redundancy,
                technology_redundancy=technology_redundancy,
                overall_redundancy=overall_redundancy
            )
            
            redundancy_results[node] = analysis
        
        self.redundancy_analysis = redundancy_results
        return redundancy_results
    
    def perform_stress_test(self, test_type: StressTestType, **test_parameters) -> StressTestResult:
        """Perform a specific stress test on the network."""
        original_graph = self.supply_network.copy()
        
        # Apply stress test
        stressed_graph, affected_nodes = self._apply_stress_test(
            original_graph, test_type, **test_parameters
        )
        
        # Calculate resilience metrics before and after
        original_metrics = self._calculate_basic_resilience_metrics(original_graph)
        degraded_metrics = self._calculate_basic_resilience_metrics(stressed_graph)
        
        # Calculate degradation
        resilience_degradation = self._calculate_resilience_degradation(
            original_metrics, degraded_metrics
        )
        
        # Identify critical failures
        critical_failures = self._identify_critical_failures(
            stressed_graph, affected_nodes
        )
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(
            test_type, affected_nodes, **test_parameters
        )
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            test_type, affected_nodes, critical_failures
        )
        
        result = StressTestResult(
            test_type=test_type,
            test_parameters=test_parameters,
            resilience_degradation=resilience_degradation,
            affected_nodes=affected_nodes,
            critical_failures=critical_failures,
            recovery_time_estimate=recovery_time,
            mitigation_strategies=mitigation_strategies
        )
        
        self.stress_test_results.append(result)
        return result
    
    def _apply_stress_test(self, graph: nx.DiGraph, test_type: StressTestType, **parameters) -> Tuple[nx.DiGraph, List[str]]:
        """Apply stress test modifications to the graph."""
        stressed_graph = graph.copy()
        affected_nodes = []
        
        if test_type == StressTestType.NODE_REMOVAL:
            nodes_to_remove = parameters.get('nodes', [])
            if not nodes_to_remove:
                # Remove highest betweenness centrality nodes
                centrality = self.centrality_measures.get('betweenness_centrality', {})
                count = parameters.get('count', 1)
                nodes_to_remove = sorted(centrality.keys(), key=lambda x: centrality[x], reverse=True)[:count]
            
            for node in nodes_to_remove:
                if node in stressed_graph:
                    stressed_graph.remove_node(node)
                    affected_nodes.append(node)
        
        elif test_type == StressTestType.CAPACITY_REDUCTION:
            reduction_factor = parameters.get('reduction_factor', 0.5)
            target_nodes = parameters.get('nodes', list(graph.nodes()))
            
            for node in target_nodes:
                if node in stressed_graph:
                    node_data = stressed_graph.nodes[node]
                    original_capacity = node_data.get('capacity', 100)
                    node_data['capacity'] = original_capacity * reduction_factor
                    affected_nodes.append(node)
        
        elif test_type == StressTestType.GEOGRAPHIC_DISRUPTION:
            # Simulate regional disruption
            affected_region = parameters.get('region', 'Asia')
            nodes_in_region = [n for n in graph.nodes() 
                             if graph.nodes[n].get('region', 'Unknown') == affected_region]
            
            for node in nodes_in_region:
                if node in stressed_graph:
                    stressed_graph.remove_node(node)
                    affected_nodes.append(node)
        
        return stressed_graph, affected_nodes
    
    def _calculate_basic_resilience_metrics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate basic resilience metrics for a graph."""
        if graph.number_of_nodes() == 0:
            return {'connectivity': 0, 'efficiency': 0, 'robustness': 0}
        
        metrics = {}
        
        # Connectivity
        undirected_g = graph.to_undirected()
        if undirected_g.number_of_nodes() > 1:
            metrics['connectivity'] = nx.node_connectivity(undirected_g)
        else:
            metrics['connectivity'] = 0
        
        # Efficiency (inverse of average path length)
        try:
            if nx.is_connected(undirected_g) and graph.number_of_nodes() > 1:
                avg_path = nx.average_shortest_path_length(undirected_g)
                metrics['efficiency'] = 1.0 / avg_path if avg_path > 0 else 0
            else:
                metrics['efficiency'] = 0
        except:
            metrics['efficiency'] = 0
        
        # Robustness (based on degree distribution)
        degrees = [d for n, d in graph.degree()]
        if degrees:
            metrics['robustness'] = np.mean(degrees) / max(1, max(degrees))
        else:
            metrics['robustness'] = 0
        
        return metrics
    
    def _calculate_resilience_degradation(self, original: Dict[str, float], degraded: Dict[str, float]) -> float:
        """Calculate overall resilience degradation."""
        degradations = []
        
        for metric in original.keys():
            original_val = original[metric]
            degraded_val = degraded.get(metric, 0)
            
            if original_val > 0:
                degradation = (original_val - degraded_val) / original_val
            else:
                degradation = 0
            
            degradations.append(max(0, degradation))
        
        return np.mean(degradations) if degradations else 0
    
    def _identify_critical_failures(self, stressed_graph: nx.DiGraph, affected_nodes: List[str]) -> List[str]:
        """Identify critical failure points in the stressed network."""
        critical_failures = []
        
        # Check for disconnected components
        undirected_g = stressed_graph.to_undirected()
        if not nx.is_connected(undirected_g):
            components = list(nx.connected_components(undirected_g))
            if len(components) > 1:
                # Find isolated small components
                for component in components:
                    if len(component) < 0.1 * stressed_graph.number_of_nodes():
                        critical_failures.extend(list(component))
        
        # Check for nodes with no inputs or outputs
        for node in stressed_graph.nodes():
            if stressed_graph.in_degree(node) == 0 and stressed_graph.out_degree(node) == 0:
                critical_failures.append(node)
        
        return list(set(critical_failures))
    
    def _estimate_recovery_time(self, test_type: StressTestType, affected_nodes: List[str], **parameters) -> int:
        """Estimate recovery time in days."""
        base_recovery_times = {
            StressTestType.NODE_REMOVAL: 90,
            StressTestType.CAPACITY_REDUCTION: 30,
            StressTestType.GEOGRAPHIC_DISRUPTION: 180,
            StressTestType.SUPPLIER_CONCENTRATION: 120
        }
        
        base_time = base_recovery_times.get(test_type, 60)
        
        # Scale by number of affected nodes
        scaling_factor = 1 + (len(affected_nodes) * 0.1)
        
        return int(base_time * scaling_factor)
    
    def _generate_mitigation_strategies(self, test_type: StressTestType, affected_nodes: List[str], 
                                      critical_failures: List[str]) -> List[str]:
        """Generate mitigation strategies based on test results."""
        strategies = []
        
        if test_type == StressTestType.NODE_REMOVAL:
            strategies.extend([
                "Develop alternative supplier relationships",
                "Increase inventory buffers for critical components",
                "Implement demand forecasting improvements"
            ])
        
        elif test_type == StressTestType.CAPACITY_REDUCTION:
            strategies.extend([
                "Invest in capacity expansion at alternative sites",
                "Develop flexible manufacturing capabilities",
                "Implement demand management strategies"
            ])
        
        elif test_type == StressTestType.GEOGRAPHIC_DISRUPTION:
            strategies.extend([
                "Diversify supply base geographically",
                "Establish regional backup facilities",
                "Develop rapid response capabilities"
            ])
        
        # Add strategies based on critical failures
        if critical_failures:
            strategies.extend([
                "Strengthen connections to isolated nodes",
                "Develop redundant pathways for critical flows",
                "Implement early warning systems"
            ])
        
        return strategies
    
    def generate_resilience_recommendations(self) -> Dict[str, Any]:
        """Generate comprehensive resilience improvement recommendations."""
        recommendations = {
            'priority_actions': [],
            'infrastructure_improvements': [],
            'risk_mitigation': [],
            'investment_priorities': [],
            'monitoring_recommendations': []
        }
        
        # Analyze current resilience state
        structural_score = self.resilience_scores.get(ResilienceMetric.STRUCTURAL_RESILIENCE)
        functional_score = self.resilience_scores.get(ResilienceMetric.FUNCTIONAL_RESILIENCE)
        
        # Priority actions based on scores
        if structural_score and structural_score.score < 0.6:
            recommendations['priority_actions'].extend([
                "Improve network connectivity and redundancy",
                "Address structural vulnerabilities immediately",
                "Conduct comprehensive network redesign"
            ])
        
        if functional_score and functional_score.score < 0.6:
            recommendations['priority_actions'].extend([
                "Increase capacity buffers at critical nodes",
                "Optimize flow distribution and load balancing",
                "Implement capacity management systems"
            ])
        
        # Infrastructure improvements
        critical_nodes = self._identify_most_critical_nodes()
        for node_info in critical_nodes[:3]:  # Top 3 critical nodes
            recommendations['infrastructure_improvements'].append(
                f"Strengthen resilience at {node_info['node']} (criticality: {node_info['criticality']:.2f})"
            )
        
        # Risk mitigation
        recommendations['risk_mitigation'].extend([
            "Develop supplier diversification strategy",
            "Establish geographic distribution of critical capabilities",
            "Create rapid response and recovery procedures",
            "Implement real-time monitoring and alerting"
        ])
        
        # Investment priorities
        geographic_risk = self._assess_geographic_concentration()
        supplier_risk = self._assess_supplier_concentration()
        
        if max(geographic_risk.values()) > 0.7:
            recommendations['investment_priorities'].append(
                "High priority: Geographic diversification of supply base"
            )
        
        if max(supplier_risk.values()) > 0.7:
            recommendations['investment_priorities'].append(
                "High priority: Supplier base diversification"
            )
        
        # Monitoring recommendations
        recommendations['monitoring_recommendations'].extend([
            "Implement continuous network health monitoring",
            "Establish resilience KPIs and regular assessment",
            "Create stress testing schedule (quarterly)",
            "Monitor supplier performance and capacity utilization"
        ])
        
        return recommendations
    
    def get_resilience_summary(self) -> Dict[str, Any]:
        """Get comprehensive resilience analysis summary."""
        summary = {
            'overall_assessment': {},
            'resilience_scores': {},
            'key_vulnerabilities': [],
            'critical_nodes': [],
            'recommendations': [],
            'stress_test_summary': {}
        }
        
        # Overall assessment
        scores = []
        if ResilienceMetric.STRUCTURAL_RESILIENCE in self.resilience_scores:
            scores.append(self.resilience_scores[ResilienceMetric.STRUCTURAL_RESILIENCE].score)
        if ResilienceMetric.FUNCTIONAL_RESILIENCE in self.resilience_scores:
            scores.append(self.resilience_scores[ResilienceMetric.FUNCTIONAL_RESILIENCE].score)
        
        if scores:
            overall_score = np.mean(scores)
            summary['overall_assessment'] = {
                'overall_resilience_score': overall_score,
                'resilience_level': 'High' if overall_score >= 0.7 else 'Medium' if overall_score >= 0.5 else 'Low',
                'network_health': 'Good' if overall_score >= 0.6 else 'Needs Improvement'
            }
        
        # Resilience scores
        for metric, score in self.resilience_scores.items():
            summary['resilience_scores'][metric.value] = {
                'score': score.score,
                'interpretation': score.interpretation,
                'components': score.components
            }
        
        # Key vulnerabilities
        critical_nodes = self._identify_most_critical_nodes()
        summary['critical_nodes'] = critical_nodes[:5]  # Top 5
        
        # Stress test summary
        if self.stress_test_results:
            avg_degradation = np.mean([r.resilience_degradation for r in self.stress_test_results])
            summary['stress_test_summary'] = {
                'tests_conducted': len(self.stress_test_results),
                'average_resilience_degradation': avg_degradation,
                'max_recovery_time': max([r.recovery_time_estimate for r in self.stress_test_results])
            }
        
        return summary
    
    def _identify_most_critical_nodes(self) -> List[Dict[str, Any]]:
        """Identify the most critical nodes based on multiple criteria."""
        critical_nodes = []
        
        for node in self.supply_network.nodes():
            # Combine multiple centrality measures
            criticality_factors = []
            
            for measure_name, centralities in self.centrality_measures.items():
                if node in centralities:
                    criticality_factors.append(centralities[node])
            
            # Add redundancy inverse (less redundancy = more critical)
            if node in self.redundancy_analysis:
                redundancy = self.redundancy_analysis[node].overall_redundancy
                criticality_factors.append(1.0 - redundancy)
            
            overall_criticality = np.mean(criticality_factors) if criticality_factors else 0
            
            critical_nodes.append({
                'node': node,
                'criticality': overall_criticality,
                'factors': {
                    'betweenness_centrality': self.centrality_measures.get('betweenness_centrality', {}).get(node, 0),
                    'degree_centrality': self.centrality_measures.get('degree_centrality', {}).get(node, 0),
                    'redundancy_score': self.redundancy_analysis.get(node, type('obj', (object,), {'overall_redundancy': 0})).overall_redundancy
                }
            })
        
        return sorted(critical_nodes, key=lambda x: x['criticality'], reverse=True)
    
    def _assess_geographic_concentration(self) -> Dict[str, float]:
        """Assess geographic concentration risk."""
        region_counts = defaultdict(int)
        total_nodes = self.supply_network.number_of_nodes()
        
        for node in self.supply_network.nodes():
            region = self.supply_network.nodes[node].get('region', 'Unknown')
            region_counts[region] += 1
        
        # Calculate concentration ratios
        concentration_ratios = {}
        for region, count in region_counts.items():
            concentration_ratios[region] = count / total_nodes if total_nodes > 0 else 0
        
        return concentration_ratios
    
    def _assess_supplier_concentration(self) -> Dict[str, float]:
        """Assess supplier concentration risk."""
        supplier_counts = defaultdict(int)
        total_supply_relationships = 0
        
                 for node in self.supply_network.nodes():
             supplier_count = int(self.supply_network.in_degree(node))
             if supplier_count > 0:
                 supplier_counts[node] = supplier_count
                 total_supply_relationships += supplier_count
        
        # Calculate concentration based on supplier diversity
        concentration_ratios = {}
        for node, count in supplier_counts.items():
            # Higher concentration = fewer suppliers relative to total
            concentration_ratios[node] = 1.0 - (count / 10.0)  # Normalize by ideal supplier count
        
        return concentration_ratios
