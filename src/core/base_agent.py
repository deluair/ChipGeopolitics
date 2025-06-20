"""
Base agent class for ChipGeopolitics simulation.
All agent types inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import uuid
from mesa import Agent
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Enumeration of agent types in the simulation."""
    HYPERSCALER = "hyperscaler"
    CHIP_MANUFACTURER = "chip_manufacturer"
    EQUIPMENT_SUPPLIER = "equipment_supplier"
    NATION_STATE = "nation_state"


@dataclass
class AgentMetrics:
    """Container for agent performance metrics."""
    revenue: float = 0.0
    costs: float = 0.0
    profit_margin: float = 0.0
    market_share: float = 0.0
    capacity_utilization: float = 0.0
    inventory_levels: float = 0.0
    supply_chain_resilience: float = 0.0
    geopolitical_risk_exposure: float = 0.0


class BaseAgent(Agent, ABC):
    """
    Base class for all agents in the ChipGeopolitics simulation.
    
    Provides common functionality for:
    - Economic decision making
    - Supply chain interactions
    - Geopolitical risk assessment
    - Market dynamics participation
    """
    
    def __init__(self, unique_id: int, model, agent_type: AgentType, 
                 name: str, initial_capital: float = 0.0):
        """
        Initialize base agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: Reference to the simulation model
            agent_type: Type of agent (hyperscaler, manufacturer, etc.)
            name: Human-readable name for the agent
            initial_capital: Starting capital for the agent
        """
        super().__init__(unique_id, model)
        
        self.agent_type = agent_type
        self.name = name
        self.agent_uuid = str(uuid.uuid4())
        
        # Economic attributes
        self.capital = initial_capital
        self.revenue_history: List[float] = []
        self.cost_history: List[float] = []
        
        # Performance metrics
        self.metrics = AgentMetrics()
        
        # Strategic attributes
        self.risk_tolerance = 0.5  # 0-1 scale
        self.innovation_index = 0.5  # 0-1 scale
        self.geopolitical_stance = 0.5  # 0-1 scale (0=isolationist, 1=globalist)
        
        # Supply chain connections
        self.suppliers: List['BaseAgent'] = []
        self.customers: List['BaseAgent'] = []
        self.strategic_partners: List['BaseAgent'] = []
        
        # Location and regional attributes
        self.primary_region = "Global"
        self.regional_presence: Dict[str, float] = {}  # region -> presence strength
        
        # Market position
        self.market_position_history: List[Dict[str, Any]] = []
        self.competitive_advantages: List[str] = []
        
    @abstractmethod
    def step(self) -> None:
        """
        Execute one simulation step for this agent.
        Must be implemented by all subclasses.
        """
        pass
    
    @abstractmethod
    def make_strategic_decisions(self) -> Dict[str, Any]:
        """
        Make strategic decisions based on current market conditions.
        Must be implemented by all subclasses.
        
        Returns:
            Dictionary containing strategic decisions
        """
        pass
    
    @abstractmethod
    def assess_geopolitical_risk(self) -> float:
        """
        Assess current geopolitical risk exposure.
        Must be implemented by all subclasses.
        
        Returns:
            Risk assessment score (0-1 scale)
        """
        pass
    
    def update_metrics(self) -> None:
        """Update agent performance metrics."""
        # Calculate basic financial metrics
        if len(self.revenue_history) > 0:
            current_revenue = self.revenue_history[-1]
            current_costs = self.cost_history[-1] if len(self.cost_history) > 0 else 0
            
            self.metrics.revenue = current_revenue
            self.metrics.costs = current_costs
            self.metrics.profit_margin = (current_revenue - current_costs) / max(current_revenue, 1)
        
        # Update geopolitical risk exposure
        self.metrics.geopolitical_risk_exposure = self.assess_geopolitical_risk()
    
    def add_supplier(self, supplier: 'BaseAgent', relationship_strength: float = 1.0) -> None:
        """
        Add a supplier relationship.
        
        Args:
            supplier: The supplier agent
            relationship_strength: Strength of the relationship (0-1)
        """
        if supplier not in self.suppliers:
            self.suppliers.append(supplier)
            if self not in supplier.customers:
                supplier.customers.append(self)
    
    def add_customer(self, customer: 'BaseAgent', relationship_strength: float = 1.0) -> None:
        """
        Add a customer relationship.
        
        Args:
            customer: The customer agent
            relationship_strength: Strength of the relationship (0-1)
        """
        if customer not in self.customers:
            self.customers.append(customer)
            if self not in customer.suppliers:
                customer.suppliers.append(self)
    
    def calculate_supply_chain_resilience(self) -> float:
        """
        Calculate supply chain resilience based on diversification and redundancy.
        
        Returns:
            Resilience score (0-1 scale)
        """
        if not self.suppliers:
            return 0.0
        
        # Base resilience on supplier diversification
        supplier_regions = set()
        for supplier in self.suppliers:
            supplier_regions.add(supplier.primary_region)
        
        geographic_diversification = len(supplier_regions) / max(len(self.suppliers), 1)
        supplier_count_factor = min(len(self.suppliers) / 5, 1.0)  # Optimal around 5 suppliers
        
        resilience = (geographic_diversification + supplier_count_factor) / 2
        return min(resilience, 1.0)
    
    def get_market_intelligence(self) -> Dict[str, Any]:
        """
        Gather market intelligence from the simulation model.
        
        Returns:
            Dictionary containing market intelligence data
        """
        # Get market conditions from the model
        current_step = 0
        total_agents = 0
        
        if self.model and hasattr(self.model, 'schedule') and self.model.schedule:
            current_step = getattr(self.model.schedule, 'steps', 0)
            total_agents = len(getattr(self.model.schedule, 'agents', []))
        
        market_data = {
            "current_step": current_step,
            "total_agents": total_agents,
            "market_volatility": getattr(self.model, 'market_volatility', 0.1) if self.model else 0.1,
            "geopolitical_tension": getattr(self.model, 'geopolitical_tension', 0.1) if self.model else 0.1
        }
        
        # Add agent-specific market position
        market_data["supply_chain_resilience"] = self.calculate_supply_chain_resilience()
        market_data["regional_exposure"] = self.regional_presence.copy()
        
        return market_data
    
    def record_market_position(self) -> None:
        """Record current market position for historical tracking."""
        current_step = 0
        if self.model and hasattr(self.model, 'schedule') and self.model.schedule:
            current_step = getattr(self.model.schedule, 'steps', 0)
            
        position = {
            "step": current_step,
            "revenue": self.metrics.revenue,
            "market_share": self.metrics.market_share,
            "capacity_utilization": self.metrics.capacity_utilization,
            "supply_chain_resilience": self.metrics.supply_chain_resilience,
            "geopolitical_risk": self.metrics.geopolitical_risk_exposure,
            "capital": self.capital
        }
        self.market_position_history.append(position)
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's current state.
        
        Returns:
            Dictionary containing agent summary
        """
        return {
            "id": self.unique_id,
            "uuid": self.agent_uuid,
            "name": self.name,
            "type": self.agent_type.value,
            "capital": self.capital,
            "metrics": {
                "revenue": self.metrics.revenue,
                "profit_margin": self.metrics.profit_margin,
                "market_share": self.metrics.market_share,
                "capacity_utilization": self.metrics.capacity_utilization,
                "supply_chain_resilience": self.metrics.supply_chain_resilience,
                "geopolitical_risk_exposure": self.metrics.geopolitical_risk_exposure
            },
            "strategic_attributes": {
                "risk_tolerance": self.risk_tolerance,
                "innovation_index": self.innovation_index,
                "geopolitical_stance": self.geopolitical_stance
            },
            "relationships": {
                "suppliers_count": len(self.suppliers),
                "customers_count": len(self.customers),
                "partners_count": len(self.strategic_partners)
            },
            "regional_presence": self.regional_presence
        } 