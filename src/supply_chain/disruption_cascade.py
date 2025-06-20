"""
Disruption Cascade Modeling

Models how disruptions propagate through semiconductor supply chains with
realistic cascade effects, recovery dynamics, and adaptive responses.
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import deque

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class DisruptionType(Enum):
    """Types of supply chain disruptions."""
    NATURAL_DISASTER = "natural_disaster"
    GEOPOLITICAL_EVENT = "geopolitical_event"
    CYBERATTACK = "cyberattack"
    EQUIPMENT_FAILURE = "equipment_failure"
    LABOR_STRIKE = "labor_strike"
    MATERIAL_SHORTAGE = "material_shortage"
    TRANSPORTATION_DISRUPTION = "transportation_disruption"
    REGULATORY_CHANGE = "regulatory_change"
    FINANCIAL_CRISIS = "financial_crisis"
    PANDEMIC = "pandemic"

class DisruptionSeverity(Enum):
    """Disruption severity levels."""
    MINOR = "minor"          # <10% capacity impact
    MODERATE = "moderate"    # 10-30% capacity impact  
    MAJOR = "major"         # 30-70% capacity impact
    SEVERE = "severe"       # 70-90% capacity impact
    CATASTROPHIC = "catastrophic"  # >90% capacity impact

class RecoveryPhase(Enum):
    """Recovery phase types."""
    IMMEDIATE_RESPONSE = "immediate_response"    # 0-7 days
    SHORT_TERM_RECOVERY = "short_term_recovery"  # 7-30 days
    MEDIUM_TERM_RECOVERY = "medium_term_recovery" # 30-180 days
    LONG_TERM_RECOVERY = "long_term_recovery"    # 180+ days
    FULL_RECOVERY = "full_recovery"

@dataclass
class DisruptionEvent:
    """Disruption event definition."""
    event_id: str
    disruption_type: DisruptionType
    severity: DisruptionSeverity
    affected_nodes: List[str]
    start_time: int  # Time step
    duration_estimate: int  # Expected duration in time steps
    capacity_impact: Dict[str, float]  # Node-specific capacity reduction (0-1)
    geographic_scope: List[str]  # Affected regions
    probability_spread: float  # Probability of spreading to connected nodes
    recovery_curve: str  # "linear", "exponential", "s_curve"
    
@dataclass
class CascadeState:
    """State of cascade propagation at a given time."""
    time_step: int
    affected_nodes: Set[str]
    capacity_reductions: Dict[str, float]
    cumulative_impact: float
    active_disruptions: List[str]
    recovery_progress: Dict[str, float]  # Recovery progress by node (0-1)
    
@dataclass
class AdaptiveResponse:
    """Adaptive response to disruptions."""
    response_id: str
    trigger_conditions: Dict[str, float]
    response_type: str  # "rerouting", "inventory_release", "capacity_shift", "supplier_switch"
    target_nodes: List[str]
    effectiveness: float  # 0-1
    implementation_time: int  # Time steps to implement
    cost: float  # Implementation cost
    
class DisruptionCascadeModel:
    """
    Advanced disruption cascade modeling for semiconductor supply chains.
    
    Simulates:
    - Multi-node disruption propagation
    - Realistic recovery dynamics
    - Adaptive supply chain responses
    - Inventory buffer effects
    - Alternative route activation
    - Economic impact assessment
    """
    
    def __init__(self, supply_network: nx.DiGraph, time_horizon: int = 365):
        self.supply_network = supply_network
        self.time_horizon = time_horizon
        
        # Simulation state
        self.current_time = 0
        self.cascade_history: List[CascadeState] = []
        self.active_events: List[DisruptionEvent] = []
        
        # Node properties
        self.node_capacities: Dict[str, float] = {}
        self.node_inventories: Dict[str, float] = {}
        self.node_resilience: Dict[str, float] = {}
        self.recovery_rates: Dict[str, float] = {}
        
        # Adaptive responses
        self.adaptive_responses: List[AdaptiveResponse] = []
        self.response_triggers: Dict[str, List[AdaptiveResponse]] = {}
        
        # Economic impact tracking
        self.economic_losses: List[float] = []
        self.recovery_costs: List[float] = []
        
        self._initialize_node_properties()
        self._initialize_adaptive_responses()
    
    def _initialize_node_properties(self):
        """Initialize node properties from supply network."""
        for node_id in self.supply_network.nodes():
            node_data = self.supply_network.nodes[node_id]
            
            # Extract or estimate node properties
            self.node_capacities[node_id] = node_data.get('capacity', 10000)
            self.node_inventories[node_id] = node_data.get('capacity', 10000) * 0.15  # 15% inventory buffer
            self.node_resilience[node_id] = node_data.get('reliability', 0.85)
            
            # Recovery rate based on node type and characteristics
            node_type = node_data.get('node_type', 'component_supplier')
            if 'foundry' in node_type.lower():
                self.recovery_rates[node_id] = 0.02  # Slow recovery for complex fabs
            elif 'equipment' in node_type.lower():
                self.recovery_rates[node_id] = 0.01  # Very slow for equipment suppliers
            elif 'materials' in node_type.lower():
                self.recovery_rates[node_id] = 0.05  # Faster for materials
            else:
                self.recovery_rates[node_id] = 0.03  # Moderate for other nodes
    
    def _initialize_adaptive_responses(self):
        """Initialize adaptive response mechanisms."""
        
        # Inventory release response
        inventory_response = AdaptiveResponse(
            response_id="inventory_release",
            trigger_conditions={"capacity_reduction": 0.3, "downstream_demand": 0.8},
            response_type="inventory_release",
            target_nodes=[],  # Applied dynamically
            effectiveness=0.6,
            implementation_time=2,  # 2 time steps
            cost=0.1  # 10% of normal operating cost
        )
        
        # Supply rerouting response
        rerouting_response = AdaptiveResponse(
            response_id="supply_rerouting",
            trigger_conditions={"capacity_reduction": 0.5, "alternative_paths": 1},
            response_type="rerouting",
            target_nodes=[],
            effectiveness=0.4,
            implementation_time=5,
            cost=0.2
        )
        
        # Supplier switching response
        supplier_switch_response = AdaptiveResponse(
            response_id="supplier_switch",
            trigger_conditions={"capacity_reduction": 0.7, "alternative_suppliers": 1},
            response_type="supplier_switch",
            target_nodes=[],
            effectiveness=0.7,
            implementation_time=10,
            cost=0.3
        )
        
        # Emergency capacity activation
        capacity_activation_response = AdaptiveResponse(
            response_id="emergency_capacity",
            trigger_conditions={"capacity_reduction": 0.4, "spare_capacity": 0.1},
            response_type="capacity_shift",
            target_nodes=[],
            effectiveness=0.5,
            implementation_time=7,
            cost=0.5
        )
        
        self.adaptive_responses = [
            inventory_response,
            rerouting_response,
            supplier_switch_response,
            capacity_activation_response
        ]
    
    def simulate_disruption_cascade(self, initial_disruption: DisruptionEvent, 
                                  enable_adaptation: bool = True) -> List[CascadeState]:
        """Simulate disruption cascade propagation over time."""
        
        # Reset simulation state
        self.current_time = 0
        self.cascade_history = []
        self.active_events = [initial_disruption]
        self.economic_losses = []
        self.recovery_costs = []
        
        # Initialize cascade state
        initial_state = CascadeState(
            time_step=0,
            affected_nodes=set(initial_disruption.affected_nodes),
            capacity_reductions=initial_disruption.capacity_impact.copy(),
            cumulative_impact=sum(initial_disruption.capacity_impact.values()),
            active_disruptions=[initial_disruption.event_id],
            recovery_progress={}
        )
        self.cascade_history.append(initial_state)
        
        # Run simulation
        for t in range(1, self.time_horizon):
            self.current_time = t
            
            # Update existing disruptions
            current_state = self._update_disruption_state(self.cascade_history[-1])
            
            # Propagate disruptions to connected nodes
            current_state = self._propagate_disruptions(current_state)
            
            # Apply adaptive responses if enabled
            if enable_adaptation:
                current_state = self._apply_adaptive_responses(current_state)
            
            # Update recovery progress
            current_state = self._update_recovery_progress(current_state)
            
            # Calculate economic impact
            economic_loss = self._calculate_economic_impact(current_state)
            self.economic_losses.append(economic_loss)
            
            # Store state
            self.cascade_history.append(current_state)
            
            # Check if cascade has ended
            if not current_state.active_disruptions and not current_state.affected_nodes:
                break
        
        return self.cascade_history
    
    def _update_disruption_state(self, previous_state: CascadeState) -> CascadeState:
        """Update state of active disruptions."""
        
        new_state = CascadeState(
            time_step=self.current_time,
            affected_nodes=previous_state.affected_nodes.copy(),
            capacity_reductions=previous_state.capacity_reductions.copy(),
            cumulative_impact=previous_state.cumulative_impact,
            active_disruptions=previous_state.active_disruptions.copy(),
            recovery_progress=previous_state.recovery_progress.copy()
        )
        
        # Update active events
        events_to_remove = []
        for event in self.active_events:
            event_age = self.current_time - event.start_time
            
            # Check if event duration has passed
            if event_age >= event.duration_estimate:
                events_to_remove.append(event)
                if event.event_id in new_state.active_disruptions:
                    new_state.active_disruptions.remove(event.event_id)
        
        # Remove expired events
        for event in events_to_remove:
            self.active_events.remove(event)
        
        return new_state
    
    def _propagate_disruptions(self, current_state: CascadeState) -> CascadeState:
        """Propagate disruptions to connected nodes."""
        
        new_affected = set()
        new_capacity_reductions = {}
        
        # Check each currently affected node for propagation
        for affected_node in current_state.affected_nodes:
            capacity_reduction = current_state.capacity_reductions.get(affected_node, 0)
            
            # Get downstream nodes
            downstream_nodes = list(self.supply_network.successors(affected_node))
            
            for downstream_node in downstream_nodes:
                if downstream_node not in current_state.affected_nodes:
                    # Calculate propagation probability
                    prop_prob = self._calculate_propagation_probability(
                        affected_node, downstream_node, capacity_reduction
                    )
                    
                    # Stochastic propagation
                    if np.random.random() < prop_prob:
                        new_affected.add(downstream_node)
                        
                        # Calculate downstream impact
                        downstream_impact = self._calculate_downstream_impact(
                            affected_node, downstream_node, capacity_reduction
                        )
                        new_capacity_reductions[downstream_node] = downstream_impact
        
        # Update state with new affected nodes
        current_state.affected_nodes.update(new_affected)
        current_state.capacity_reductions.update(new_capacity_reductions)
        current_state.cumulative_impact = sum(current_state.capacity_reductions.values())
        
        return current_state
    
    def _calculate_propagation_probability(self, source_node: str, target_node: str, 
                                         source_impact: float) -> float:
        """Calculate probability of disruption propagating between nodes."""
        
        # Base propagation probability
        base_prob = 0.1
        
        # Impact factor (higher source impact = higher propagation probability)
        impact_factor = source_impact ** 0.5
        
        # Edge weight factor (stronger connections = higher propagation)
        edge_data = self.supply_network.get_edge_data(source_node, target_node, {})
        edge_strength = edge_data.get('reliability', 0.5)
        
        # Target resilience factor (more resilient nodes less likely to be affected)
        target_resilience = self.node_resilience.get(target_node, 0.85)
        resilience_factor = 1 - target_resilience
        
        # Inventory buffer factor (more inventory = lower propagation probability)
        target_inventory = self.node_inventories.get(target_node, 0)
        target_capacity = self.node_capacities.get(target_node, 1)
        inventory_ratio = target_inventory / target_capacity if target_capacity > 0 else 0
        inventory_factor = max(0.2, 1 - inventory_ratio)  # Min 20% probability even with inventory
        
        # Combined probability
        propagation_prob = (base_prob * 
                          impact_factor * 
                          edge_strength * 
                          resilience_factor * 
                          inventory_factor)
        
        return min(0.8, propagation_prob)  # Cap at 80%
    
    def _calculate_downstream_impact(self, source_node: str, target_node: str, 
                                   source_impact: float) -> float:
        """Calculate impact on downstream node from upstream disruption."""
        
        # Base impact attenuation
        base_attenuation = 0.7  # Downstream impact is typically less than upstream
        
        # Dependency factor (how much target depends on source)
        source_predecessors = list(self.supply_network.predecessors(target_node))
        dependency_factor = 1.0 / len(source_predecessors) if source_predecessors else 1.0
        
        # Edge reliability factor
        edge_data = self.supply_network.get_edge_data(source_node, target_node, {})
        reliability_factor = edge_data.get('reliability', 0.8)
        
        # Inventory buffer mitigation
        target_inventory = self.node_inventories.get(target_node, 0)
        target_capacity = self.node_capacities.get(target_node, 1)
        inventory_ratio = target_inventory / target_capacity if target_capacity > 0 else 0
        inventory_mitigation = max(0.3, 1 - inventory_ratio * 2)  # Inventory can reduce impact by up to 70%
        
        # Calculate final impact
        downstream_impact = (source_impact * 
                           base_attenuation * 
                           dependency_factor * 
                           reliability_factor * 
                           inventory_mitigation)
        
        return min(1.0, downstream_impact)
    
    def _apply_adaptive_responses(self, current_state: CascadeState) -> CascadeState:
        """Apply adaptive responses to mitigate disruption impacts."""
        
        for response in self.adaptive_responses:
            # Check trigger conditions
            if self._check_response_triggers(response, current_state):
                # Apply response
                current_state = self._execute_response(response, current_state)
                
                # Add implementation cost
                implementation_cost = response.cost * sum(current_state.capacity_reductions.values()) * 1000
                self.recovery_costs.append(implementation_cost)
        
        return current_state
    
    def _check_response_triggers(self, response: AdaptiveResponse, state: CascadeState) -> bool:
        """Check if response trigger conditions are met."""
        
        # Check capacity reduction threshold
        max_capacity_reduction = max(state.capacity_reductions.values()) if state.capacity_reductions else 0
        capacity_threshold = response.trigger_conditions.get("capacity_reduction", 1.0)
        
        if max_capacity_reduction < capacity_threshold:
            return False
        
        # Additional checks based on response type
        if response.response_type == "rerouting":
            # Check if alternative paths exist
            affected_nodes = list(state.affected_nodes)
            if affected_nodes:
                alternatives = self._count_alternative_paths(affected_nodes[0])
                if alternatives < response.trigger_conditions.get("alternative_paths", 1):
                    return False
        
        elif response.response_type == "supplier_switch":
            # Check if alternative suppliers exist
            if len(state.affected_nodes) == 0:
                return False
            # Simplified check - assume alternatives exist if multiple suppliers in network
            total_suppliers = len([n for n in self.supply_network.nodes() 
                                 if 'supplier' in str(self.supply_network.nodes[n].get('node_type', ''))])
            if total_suppliers < 3:
                return False
        
        return True
    
    def _execute_response(self, response: AdaptiveResponse, state: CascadeState) -> CascadeState:
        """Execute adaptive response and update state."""
        
        if response.response_type == "inventory_release":
            # Reduce capacity impact by releasing inventory
            for node in state.affected_nodes:
                if node in state.capacity_reductions:
                    current_reduction = state.capacity_reductions[node]
                    inventory_benefit = response.effectiveness * 0.3  # Max 30% reduction
                    new_reduction = max(0, current_reduction - inventory_benefit)
                    state.capacity_reductions[node] = new_reduction
                    
                    # Reduce inventory
                    current_inventory = self.node_inventories.get(node, 0)
                    self.node_inventories[node] = max(0, current_inventory * 0.7)
        
        elif response.response_type == "rerouting":
            # Find alternative paths and reduce impact
            for node in list(state.affected_nodes):
                if node in state.capacity_reductions:
                    alternatives = self._count_alternative_paths(node)
                    if alternatives > 0:
                        reduction_benefit = response.effectiveness * min(0.5, alternatives * 0.1)
                        current_reduction = state.capacity_reductions[node]
                        state.capacity_reductions[node] = max(0, current_reduction - reduction_benefit)
        
        elif response.response_type == "supplier_switch":
            # Switch to alternative suppliers (reduce impact over time)
            for node in list(state.affected_nodes):
                if node in state.capacity_reductions:
                    current_reduction = state.capacity_reductions[node]
                    switch_benefit = response.effectiveness * 0.4  # Max 40% reduction
                    state.capacity_reductions[node] = max(0, current_reduction - switch_benefit)
        
        elif response.response_type == "capacity_shift":
            # Activate emergency capacity
            for node in state.affected_nodes:
                if node in state.capacity_reductions:
                    current_reduction = state.capacity_reductions[node]
                    capacity_benefit = response.effectiveness * 0.25  # Max 25% reduction
                    state.capacity_reductions[node] = max(0, current_reduction - capacity_benefit)
        
        # Update cumulative impact
        state.cumulative_impact = sum(state.capacity_reductions.values())
        
        return state
    
    def _count_alternative_paths(self, node: str) -> int:
        """Count alternative paths bypassing a node."""
        try:
            # Simple heuristic: count alternative suppliers/paths
            predecessors = list(self.supply_network.predecessors(node))
            successors = list(self.supply_network.successors(node))
            
            # Create temporary network without this node
            temp_network = self.supply_network.copy()
            temp_network.remove_node(node)
            
            alternatives = 0
            for pred in predecessors:
                for succ in successors:
                    try:
                        if nx.has_path(temp_network, pred, succ):
                            alternatives += 1
                    except:
                        continue
            
            return alternatives
        except:
            return 0
    
    def _update_recovery_progress(self, current_state: CascadeState) -> CascadeState:
        """Update recovery progress for affected nodes."""
        
        nodes_to_remove = []
        
        for node in current_state.affected_nodes:
            if node not in current_state.recovery_progress:
                current_state.recovery_progress[node] = 0.0
            
            # Calculate recovery rate based on node characteristics
            base_recovery_rate = self.recovery_rates.get(node, 0.03)
            
            # Modify recovery rate based on disruption severity
            current_impact = current_state.capacity_reductions.get(node, 0)
            severity_factor = 1 - (current_impact * 0.5)  # More severe = slower recovery
            
            # Apply recovery
            recovery_increment = base_recovery_rate * severity_factor
            current_state.recovery_progress[node] += recovery_increment
            
            # Update capacity reduction based on recovery
            if current_state.recovery_progress[node] > 0:
                recovery_benefit = current_state.recovery_progress[node] * current_impact
                new_reduction = max(0, current_impact - recovery_benefit)
                current_state.capacity_reductions[node] = new_reduction
                
                # If fully recovered, remove from affected nodes
                if new_reduction < 0.01:  # Effectively zero
                    nodes_to_remove.append(node)
        
        # Remove fully recovered nodes
        for node in nodes_to_remove:
            current_state.affected_nodes.discard(node)
            if node in current_state.capacity_reductions:
                del current_state.capacity_reductions[node]
        
        # Update cumulative impact
        current_state.cumulative_impact = sum(current_state.capacity_reductions.values())
        
        return current_state
    
    def _calculate_economic_impact(self, state: CascadeState) -> float:
        """Calculate economic impact of current disruption state."""
        
        total_economic_loss = 0.0
        
        for node, capacity_reduction in state.capacity_reductions.items():
            # Estimate daily economic value of node
            node_capacity = self.node_capacities.get(node, 10000)
            
            # Rough economic value per unit capacity (varies by node type)
            node_data = self.supply_network.nodes.get(node, {})
            node_type = node_data.get('node_type', 'component_supplier')
            
            if 'foundry' in str(node_type).lower():
                value_per_unit = 100  # $100 per unit for foundry capacity
            elif 'equipment' in str(node_type).lower():
                value_per_unit = 1000  # $1000 per unit for equipment
            elif 'oem' in str(node_type).lower():
                value_per_unit = 50   # $50 per unit for OEMs
            else:
                value_per_unit = 25   # $25 per unit for other nodes
            
            # Calculate daily loss
            daily_loss = node_capacity * capacity_reduction * value_per_unit
            total_economic_loss += daily_loss
        
        return total_economic_loss
    
    def analyze_cascade_results(self, cascade_history: List[CascadeState]) -> Dict[str, Any]:
        """Analyze cascade simulation results."""
        
        if not cascade_history:
            return {}
        
        # Basic statistics
        max_affected_nodes = max(len(state.affected_nodes) for state in cascade_history)
        peak_cumulative_impact = max(state.cumulative_impact for state in cascade_history)
        total_duration = len(cascade_history)
        
        # Recovery analysis
        recovery_time = None
        for i, state in enumerate(cascade_history):
            if state.cumulative_impact < 0.01:  # Effectively recovered
                recovery_time = i
                break
        
        # Most affected nodes
        node_impact_totals = {}
        for state in cascade_history:
            for node, impact in state.capacity_reductions.items():
                node_impact_totals[node] = node_impact_totals.get(node, 0) + impact
        
        most_affected_nodes = sorted(node_impact_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Economic impact
        total_economic_loss = sum(self.economic_losses)
        total_recovery_cost = sum(self.recovery_costs)
        
        # Propagation analysis
        initial_affected = len(cascade_history[0].affected_nodes) if cascade_history else 0
        max_propagation = max_affected_nodes - initial_affected
        
        return {
            'simulation_duration': total_duration,
            'max_affected_nodes': max_affected_nodes,
            'peak_cumulative_impact': peak_cumulative_impact,
            'recovery_time': recovery_time,
            'total_economic_loss': total_economic_loss,
            'total_recovery_cost': total_recovery_cost,
            'most_affected_nodes': most_affected_nodes,
            'propagation_extent': max_propagation,
            'average_daily_impact': np.mean([state.cumulative_impact for state in cascade_history]),
            'cascade_efficiency': peak_cumulative_impact / max(1, max_affected_nodes),  # Impact per affected node
            'recovery_efficiency': 1 / max(1, recovery_time) if recovery_time else 0,
            'adaptation_cost_ratio': total_recovery_cost / max(1, total_economic_loss)
        }
    
    def run_scenario_analysis(self, disruption_scenarios: List[DisruptionEvent], 
                            enable_adaptation: bool = True) -> Dict[str, Any]:
        """Run multiple disruption scenarios for comparative analysis."""
        
        scenario_results = {}
        
        for scenario in disruption_scenarios:
            # Run cascade simulation for this scenario
            cascade_history = self.simulate_disruption_cascade(scenario, enable_adaptation)
            
            # Analyze results
            analysis = self.analyze_cascade_results(cascade_history)
            
            scenario_results[scenario.event_id] = {
                'scenario': scenario,
                'cascade_history': cascade_history,
                'analysis': analysis
            }
        
        # Comparative analysis
        comparative_analysis = self._compare_scenarios(scenario_results)
        
        return {
            'individual_scenarios': scenario_results,
            'comparative_analysis': comparative_analysis
        }
    
    def _compare_scenarios(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results across multiple scenarios."""
        
        if not scenario_results:
            return {}
        
        # Extract key metrics for comparison
        metrics = ['peak_cumulative_impact', 'total_economic_loss', 'recovery_time', 'max_affected_nodes']
        
        comparison = {}
        for metric in metrics:
            values = []
            scenario_names = []
            
            for scenario_id, result in scenario_results.items():
                if metric in result['analysis']:
                    values.append(result['analysis'][metric])
                    scenario_names.append(scenario_id)
            
            if values:
                comparison[metric] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'worst_scenario': scenario_names[values.index(max(values))],
                    'best_scenario': scenario_names[values.index(min(values))]
                }
        
        # Risk ranking
        risk_scores = []
        for scenario_id, result in scenario_results.items():
            analysis = result['analysis']
            # Simple risk score combining multiple factors
            risk_score = (analysis.get('peak_cumulative_impact', 0) * 
                         analysis.get('total_economic_loss', 0) / 1000000 +  # Normalize economic loss
                         analysis.get('max_affected_nodes', 0) * 0.1)
            
            risk_scores.append((scenario_id, risk_score))
        
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'metric_comparisons': comparison,
            'risk_ranking': risk_scores,
            'total_scenarios': len(scenario_results),
            'summary_insights': self._generate_scenario_insights(scenario_results)
        }
    
    def _generate_scenario_insights(self, scenario_results: Dict[str, Any]) -> List[str]:
        """Generate insights from scenario analysis."""
        insights = []
        
        if len(scenario_results) < 2:
            return insights
        
        # Find most/least impactful disruption types
        disruption_impacts = {}
        for scenario_id, result in scenario_results.items():
            disruption_type = result['scenario'].disruption_type.value
            impact = result['analysis'].get('peak_cumulative_impact', 0)
            
            if disruption_type not in disruption_impacts:
                disruption_impacts[disruption_type] = []
            disruption_impacts[disruption_type].append(impact)
        
        avg_impacts = {dt: np.mean(impacts) for dt, impacts in disruption_impacts.items()}
        if avg_impacts:
            most_impactful = max(avg_impacts.items(), key=lambda x: x[1])
            least_impactful = min(avg_impacts.items(), key=lambda x: x[1])
            
            insights.append(f"Most impactful disruption type: {most_impactful[0]} (avg impact: {most_impactful[1]:.2f})")
            insights.append(f"Least impactful disruption type: {least_impactful[0]} (avg impact: {least_impactful[1]:.2f})")
        
        # Recovery time analysis
        recovery_times = [result['analysis'].get('recovery_time') for result in scenario_results.values()]
        recovery_times = [rt for rt in recovery_times if rt is not None]
        
        if recovery_times:
            avg_recovery = np.mean(recovery_times)
            insights.append(f"Average recovery time: {avg_recovery:.1f} time steps")
            
            if max(recovery_times) > avg_recovery * 2:
                insights.append("Some scenarios show significantly longer recovery times, indicating critical vulnerabilities")
        
        # Economic impact insights
        economic_losses = [result['analysis'].get('total_economic_loss', 0) for result in scenario_results.values()]
        if economic_losses:
            total_risk_exposure = sum(economic_losses)
            insights.append(f"Total economic risk exposure: ${total_risk_exposure/1000000:.1f}M across all scenarios")
        
        return insights
    
    def get_disruption_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of disruption cascade capabilities."""
        
        return {
            'supply_network_nodes': len(self.supply_network.nodes()),
            'supply_network_edges': len(self.supply_network.edges()),
            'time_horizon': self.time_horizon,
            'adaptive_responses_available': len(self.adaptive_responses),
            'disruption_types_supported': [dt.value for dt in DisruptionType],
            'severity_levels': [sl.value for sl in DisruptionSeverity],
            'recovery_phases': [rp.value for rp in RecoveryPhase],
            'node_resilience_range': (min(self.node_resilience.values()), max(self.node_resilience.values())),
            'recovery_rate_range': (min(self.recovery_rates.values()), max(self.recovery_rates.values())),
            'total_network_capacity': sum(self.node_capacities.values()),
            'total_inventory_buffer': sum(self.node_inventories.values())
        } 