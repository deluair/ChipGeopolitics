"""
Simulation Engine for ChipGeopolitics Framework

Core simulation engine that orchestrates agents, market dynamics,
and scenario execution for the semiconductor geopolitics simulation.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    max_steps: int = 100
    convergence_threshold: float = 0.01
    collect_data: bool = True
    random_seed: int = 42


class SimulationEngine:
    """
    Core simulation engine for ChipGeopolitics framework.
    
    Manages agent execution, market dynamics, and scenario progression.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize simulation engine.
        
        Args:
            config: Simulation configuration parameters
        """
        self.config = config or SimulationConfig()
        self.current_step = 0
        self.running = False
        
        # Agent management
        self.agents: Dict[str, Any] = {}
        
        # Market and economic state
        self.market_conditions = {
            "demand_growth": 0.05,
            "supply_constraints": 0.0,
            "price_volatility": 0.1,
            "geopolitical_tension": 0.2
        }
        
        # Performance tracking
        self.execution_times: List[float] = []
        self.convergence_history: List[float] = []
        self.convergence_threshold = self.config.convergence_threshold
        
        logger.info(f"Simulation engine initialized with {self.config.max_steps} max steps")
    
    def add_agent(self, agent: Any) -> None:
        """
        Add an agent to the simulation.
        
        Args:
            agent: Agent instance to add
        """
        agent_id = getattr(agent, 'agent_id', f"agent_{len(self.agents)}")
        self.agents[agent_id] = agent
        logger.debug(f"Added agent {agent_id}")
    
    def remove_agent(self, agent_id: str) -> None:
        """
        Remove an agent from the simulation.
        
        Args:
            agent_id: Agent identifier to remove
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.debug(f"Removed agent {agent_id}")
    
    def step(self) -> None:
        """Execute one simulation step."""
        step_start_time = time.time()
        
        # Update market conditions
        self._update_market_conditions()
        
        # Execute all agents
        decisions = []
        for agent in self.agents.values():
            if hasattr(agent, 'step'):
                decision = agent.step()
                if decision:
                    decisions.append(decision)
        
        # Update simulation state
        self.current_step += 1
        
        # Track performance
        step_time = time.time() - step_start_time
        self.execution_times.append(step_time)
        
        # Check convergence
        convergence_metric = self._calculate_convergence_metric(decisions)
        self.convergence_history.append(convergence_metric)
        
        # Check stopping conditions
        if self.current_step >= self.config.max_steps:
            self.running = False
            logger.info(f"Simulation completed: reached max steps ({self.config.max_steps})")
        
        if convergence_metric < self.config.convergence_threshold:
            self.running = False
            logger.info(f"Simulation converged at step {self.current_step}")
    
    def run(self, steps: Optional[int] = None, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the simulation for specified steps or until convergence.
        
        Args:
            steps: Number of steps to run
            max_steps: Maximum steps (for compatibility)
            
        Returns:
            Dictionary containing simulation results
        """
        target_steps = steps or max_steps or self.config.max_steps
        
        start_time = time.time()
        self.running = True
        logger.info(f"Starting simulation run (target steps: {target_steps})")
        
        while self.running and self.current_step < target_steps:
            self.step()
        
        # Gather final results
        results = self._compile_results()
        
        execution_time = time.time() - start_time
        logger.info(f"Simulation completed in {execution_time:.2f} seconds")
        
        return results
    
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.current_step = 0
        self.running = False
        self.execution_times = []
        self.convergence_history = []
        
        # Reset market conditions
        self.market_conditions = {
            "demand_growth": 0.05,
            "supply_constraints": 0.0,
            "price_volatility": 0.1,
            "geopolitical_tension": 0.2
        }
        
        logger.info("Simulation reset to initial state")
    
    def _update_market_conditions(self) -> None:
        """Update market conditions for current step."""
        # Simple market evolution model
        import random
        volatility_factor = 0.01
        
        # Add some randomness to market conditions
        self.market_conditions["price_volatility"] += random.uniform(-volatility_factor, volatility_factor)
        self.market_conditions["price_volatility"] = max(0.0, min(1.0, self.market_conditions["price_volatility"]))
        
        # Geopolitical tension evolves slowly
        tension_change = random.uniform(-0.005, 0.005)
        self.market_conditions["geopolitical_tension"] += tension_change
        self.market_conditions["geopolitical_tension"] = max(0.0, min(1.0, self.market_conditions["geopolitical_tension"]))
    
    def _calculate_convergence_metric(self, decisions: List[Any]) -> float:
        """
        Calculate convergence metric based on agent decisions.
        
        Args:
            decisions: List of agent decisions from current step
            
        Returns:
            Convergence metric (lower values indicate convergence)
        """
        if not decisions:
            return 0.5
        
        # Calculate utility variance as convergence metric
        utilities = []
        for decision in decisions:
            if hasattr(decision, 'utility'):
                utilities.append(decision.utility)
        
        if len(utilities) < 2:
            return 0.1
        
        # Calculate coefficient of variation
        import statistics
        mean_utility = statistics.mean(utilities)
        if mean_utility == 0:
            return 0.0
        
        std_utility = statistics.stdev(utilities)
        return std_utility / mean_utility
    
    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile final simulation results.
        
        Returns:
            Dictionary containing comprehensive simulation results
        """
        results = {
            "final_step": self.current_step,
            "final_state": {
                "agents": len(self.agents),
                "market_conditions": self.market_conditions.copy()
            },
            "converged": not self.running or (
                len(self.convergence_history) > 0 and 
                self.convergence_history[-1] < self.config.convergence_threshold
            ),
            "execution_time": sum(self.execution_times),
            "average_step_time": sum(self.execution_times) / max(len(self.execution_times), 1),
            "agent_count": len(self.agents),
            "agent_states": {}
        }
        
        # Collect final agent states
        for agent_id, agent in self.agents.items():
            results["agent_states"][agent_id] = {
                "name": getattr(agent, 'name', agent_id),
                "type": getattr(agent, 'agent_type', 'unknown'),
                "state": getattr(agent, 'state', {})
            }
        
        return results


# Additional utility classes for testing compatibility

class Decision:
    """Simple decision class for agent actions."""
    
    def __init__(self, action: str, parameters: Dict[str, Any], utility: float):
        self.action = action
        self.parameters = parameters
        self.utility = utility
        self.timestamp = datetime.now()
    
    def is_valid(self) -> bool:
        """Check if decision is valid."""
        return 0.0 <= self.utility <= 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action,
            "parameters": self.parameters,
            "utility": self.utility,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __lt__(self, other):
        return self.utility < other.utility
    
    def __gt__(self, other):
        return self.utility > other.utility


class AgentState:
    """Simple agent state management."""
    
    def __init__(self, initial_data: Dict[str, Any]):
        self.data = initial_data.copy()
        self.timestamp = datetime.now()
        self.history = []
    
    def update(self, key: str, value: Any) -> None:
        """Update state value."""
        old_value = self.data.get(key)
        self.data[key] = value
        self.timestamp = datetime.now()
        
        self.history.append({
            "timestamp": self.timestamp,
            "key": key,
            "old_value": old_value,
            "new_value": value
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get state change history."""
        return self.history.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data.copy(),
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create from dictionary."""
        state = cls(data["data"])
        state.timestamp = datetime.fromisoformat(data["timestamp"])
        return state


# For backward compatibility
class BaseAgent:
    """Simplified base agent for testing."""
    
    def __init__(self, agent_id: str, agent_type: str, initial_state: Dict[str, Any]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(initial_state)
    
    def step(self) -> Decision:
        """Execute one step and return decision."""
        return Decision("default_action", {}, 0.5)
    
    def update_state(self, key: str, value: Any) -> None:
        """Update agent state."""
        self.state.update(key, value)
    
    def validate_state(self, state_data: Dict[str, Any]) -> bool:
        """Validate state data."""
        # Simple validation - no negative resources
        for key, value in state_data.items():
            if key == "resource" and isinstance(value, (int, float)) and value < 0:
                return False
        return True
    
    def send_message(self, target_id: str, message: str, data: Dict[str, Any]) -> None:
        """Send message to another agent."""
        # Simple implementation for testing
        pass 