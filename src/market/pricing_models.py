"""
Pricing Mechanism Models

Implements sophisticated pricing models for semiconductor markets including
dynamic pricing, cost-plus models, competitive bidding, and market-based pricing.
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

class PricingMechanism(Enum):
    """Pricing mechanism types."""
    COST_PLUS = "cost_plus"
    MARKET_BASED = "market_based"
    COMPETITIVE_BIDDING = "competitive_bidding"
    LONG_TERM_CONTRACT = "long_term_contract"
    SPOT_MARKET = "spot_market"
    STRATEGIC_PRICING = "strategic_pricing"

class MarketCondition(Enum):
    """Market condition states."""
    OVERSUPPLY = "oversupply"
    BALANCED = "balanced"
    TIGHT = "tight"
    SHORTAGE = "shortage"
    CRISIS = "crisis"

@dataclass
class PricingFactors:
    """Factors affecting semiconductor pricing."""
    manufacturing_cost: float  # Base manufacturing cost per unit
    yield_rate: float  # Manufacturing yield (affects effective cost)
    demand_supply_ratio: float  # Market balance indicator
    technology_premium: float  # Premium for advanced nodes
    volume_discount: float  # Discount for large orders
    relationship_premium: float  # Premium/discount for customer relationship
    geopolitical_risk_premium: float  # Risk premium for supply security
    competitive_pressure: float  # Competitive pricing pressure
    
@dataclass
class MarketPricing:
    """Market pricing information for semiconductor products."""
    process_node: str
    product_category: str  # 'logic', 'memory', 'analog', 'mixed_signal'
    base_price_per_unit: float  # USD per unit
    pricing_mechanism: PricingMechanism
    price_elasticity: float  # Demand elasticity to price changes
    cost_structure: Dict[str, float]  # Breakdown of costs
    margin_range: Tuple[float, float]  # Min and max margin percentages
    volume_tiers: Dict[int, float]  # Volume discount tiers
    contract_premiums: Dict[str, float]  # Premium by contract type

class PricingMechanismModel:
    """
    Comprehensive pricing mechanism model for semiconductor markets.
    
    Models:
    - Dynamic pricing based on supply-demand balance
    - Cost-plus pricing with yield and volume considerations
    - Competitive pricing dynamics
    - Long-term contract pricing vs spot markets
    - Technology premiums and process node pricing
    - Customer relationship impacts on pricing
    """
    
    def __init__(self):
        # Market pricing data
        self.market_pricing: Dict[str, MarketPricing] = {}
        
        # Market condition tracking
        self.current_market_conditions: Dict[str, MarketCondition] = {}
        self.supply_demand_ratios: Dict[str, float] = {}
        self.price_history: Dict[str, List[float]] = {}
        
        # Competitive landscape
        self.competitive_intensity: Dict[str, float] = {}
        self.market_share_dynamics: Dict[str, Dict[str, float]] = {}
        
        # Initialize with realistic market data
        self._initialize_market_pricing()
        self._initialize_market_conditions()
    
    def _initialize_market_pricing(self):
        """Initialize market pricing data for different product categories."""
        # Process node pricing configurations
        pricing_configs = [
            # Leading edge nodes
            ("3nm", "logic", 15000, PricingMechanism.STRATEGIC_PRICING, -0.3, 0.4, (0.35, 0.50)),
            ("5nm", "logic", 9566, PricingMechanism.COMPETITIVE_BIDDING, -0.4, 0.45, (0.30, 0.45)),
            ("7nm", "logic", 8500, PricingMechanism.MARKET_BASED, -0.5, 0.5, (0.25, 0.40)),
            
            # Advanced nodes
            ("10nm", "logic", 6000, PricingMechanism.MARKET_BASED, -0.6, 0.55, (0.20, 0.35)),
            ("16nm", "logic", 4000, PricingMechanism.COMPETITIVE_BIDDING, -0.7, 0.6, (0.15, 0.30)),
            
            # Mature nodes
            ("28nm", "logic", 2500, PricingMechanism.COST_PLUS, -0.8, 0.7, (0.10, 0.25)),
            ("65nm", "logic", 1500, PricingMechanism.COST_PLUS, -0.9, 0.75, (0.08, 0.20)),
            
            # Memory products
            ("1x_nm", "memory", 12000, PricingMechanism.MARKET_BASED, -0.6, 0.6, (0.15, 0.35)),
            ("1y_nm", "memory", 8000, PricingMechanism.COMPETITIVE_BIDDING, -0.7, 0.65, (0.12, 0.30)),
            ("1z_nm", "memory", 5000, PricingMechanism.COST_PLUS, -0.8, 0.7, (0.10, 0.25)),
            
            # Analog and mixed signal
            ("28nm", "analog", 3000, PricingMechanism.LONG_TERM_CONTRACT, -0.4, 0.5, (0.25, 0.40)),
            ("65nm", "analog", 2000, PricingMechanism.LONG_TERM_CONTRACT, -0.3, 0.6, (0.30, 0.45)),
            ("180nm", "mixed_signal", 800, PricingMechanism.COST_PLUS, -0.5, 0.65, (0.20, 0.35)),
        ]
        
        for node, category, base_price, mechanism, elasticity, yield_rate, margin_range in pricing_configs:
            # Generate cost structure
            cost_structure = self._generate_cost_structure(node, category, base_price)
            
            # Generate volume tiers (discount by volume)
            volume_tiers = {
                1000: 1.0,      # Base price for 1K units
                10000: 0.95,    # 5% discount for 10K+
                100000: 0.90,   # 10% discount for 100K+
                1000000: 0.85,  # 15% discount for 1M+
                10000000: 0.80  # 20% discount for 10M+
            }
            
            # Contract premiums
            contract_premiums = {
                'spot': 1.1,           # 10% premium for spot market
                'short_term': 1.0,     # Base price for short-term contracts
                'long_term': 0.95,     # 5% discount for long-term contracts
                'strategic': 0.90      # 10% discount for strategic partnerships
            }
            
            pricing_key = f"{node}_{category}"
            self.market_pricing[pricing_key] = MarketPricing(
                process_node=node,
                product_category=category,
                base_price_per_unit=base_price,
                pricing_mechanism=mechanism,
                price_elasticity=elasticity,
                cost_structure=cost_structure,
                margin_range=margin_range,
                volume_tiers=volume_tiers,
                contract_premiums=contract_premiums
            )
            
            # Initialize price history
            self.price_history[pricing_key] = [base_price]
    
    def _generate_cost_structure(self, node: str, category: str, base_price: float) -> Dict[str, float]:
        """Generate realistic cost structure for semiconductor products."""
        # Base cost components (as percentage of total cost)
        if "3nm" in node or "5nm" in node:
            # Leading edge - high R&D and equipment costs
            structure = {
                'wafer_cost': 0.40,
                'packaging': 0.15,
                'testing': 0.10,
                'rd_amortization': 0.20,
                'equipment_depreciation': 0.10,
                'overhead': 0.05
            }
        elif "7nm" in node or "10nm" in node:
            # Advanced nodes
            structure = {
                'wafer_cost': 0.45,
                'packaging': 0.18,
                'testing': 0.12,
                'rd_amortization': 0.15,
                'equipment_depreciation': 0.08,
                'overhead': 0.02
            }
        else:
            # Mature nodes - lower R&D, higher manufacturing focus
            structure = {
                'wafer_cost': 0.50,
                'packaging': 0.20,
                'testing': 0.15,
                'rd_amortization': 0.08,
                'equipment_depreciation': 0.05,
                'overhead': 0.02
            }
        
        # Adjust for product category
        if category == "memory":
            structure['wafer_cost'] += 0.10  # Higher wafer cost for memory
            structure['rd_amortization'] -= 0.05
        elif category == "analog":
            structure['packaging'] += 0.05  # More complex packaging
            structure['testing'] += 0.05
            structure['wafer_cost'] -= 0.10
        
        # Convert to absolute costs
        total_cost = base_price * 0.75  # Assume 25% average margin
        absolute_costs = {component: total_cost * percentage 
                         for component, percentage in structure.items()}
        
        return absolute_costs
    
    def _initialize_market_conditions(self):
        """Initialize market condition tracking."""
        # Set initial market conditions by process node
        node_conditions = {
            "3nm": (MarketCondition.SHORTAGE, 0.7),    # High demand, limited supply
            "5nm": (MarketCondition.TIGHT, 0.85),      # Tight but improving
            "7nm": (MarketCondition.BALANCED, 1.1),    # Slightly oversupplied
            "10nm": (MarketCondition.BALANCED, 1.2),   # Well balanced
            "16nm": (MarketCondition.OVERSUPPLY, 1.4), # Oversupplied
            "28nm": (MarketCondition.OVERSUPPLY, 1.6), # Significant oversupply
            "65nm": (MarketCondition.OVERSUPPLY, 1.8)  # Mature node oversupply
        }
        
        for node, (condition, ratio) in node_conditions.items():
            self.current_market_conditions[node] = condition
            self.supply_demand_ratios[node] = ratio
        
        # Initialize competitive intensity
        self.competitive_intensity = {
            "3nm": 0.3,   # Limited competition (TSMC, Samsung)
            "5nm": 0.5,   # Moderate competition
            "7nm": 0.7,   # High competition
            "10nm": 0.8,  # Very high competition
            "16nm": 0.9,  # Intense competition
            "28nm": 0.95, # Commodity-like competition
        }
    
    def calculate_dynamic_pricing(self, product_key: str, volume: int, 
                                contract_type: str = 'short_term',
                                customer_relationship: float = 0.5,
                                custom_factors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate dynamic pricing based on market conditions and customer factors."""
        
        if product_key not in self.market_pricing:
            raise ValueError(f"Product key {product_key} not found in market pricing data")
        
        pricing_data = self.market_pricing[product_key]
        base_price = pricing_data.base_price_per_unit
        
        # Extract process node for market condition lookup
        process_node = pricing_data.process_node
        
        # 1. Volume discount
        volume_multiplier = self._calculate_volume_discount(volume, pricing_data.volume_tiers)
        
        # 2. Contract type premium/discount
        contract_multiplier = pricing_data.contract_premiums.get(contract_type, 1.0)
        
        # 3. Market condition adjustment
        market_condition = self.current_market_conditions.get(process_node, MarketCondition.BALANCED)
        supply_demand_ratio = self.supply_demand_ratios.get(process_node, 1.0)
        market_multiplier = self._calculate_market_adjustment(market_condition, supply_demand_ratio)
        
        # 4. Relationship premium/discount
        relationship_multiplier = 1.0 + (customer_relationship - 0.5) * 0.2  # ±10% based on relationship
        
        # 5. Competitive pressure adjustment
        competitive_intensity = self.competitive_intensity.get(process_node, 0.5)
        competitive_multiplier = 1.0 - (competitive_intensity * 0.15)  # Up to 15% discount in high competition
        
        # 6. Technology premium for advanced nodes
        technology_multiplier = self._calculate_technology_premium(process_node)
        
        # 7. Custom factors (geopolitical risk, supply chain disruption, etc.)
        custom_multiplier = 1.0
        if custom_factors:
            geopolitical_risk = custom_factors.get('geopolitical_risk', 0.0)
            supply_chain_risk = custom_factors.get('supply_chain_risk', 0.0)
            technology_risk = custom_factors.get('technology_risk', 0.0)
            
            custom_multiplier = (1.0 + geopolitical_risk * 0.3 + 
                               supply_chain_risk * 0.2 + 
                               technology_risk * 0.1)
        
        # Calculate final price
        final_price = (base_price * 
                      volume_multiplier * 
                      contract_multiplier * 
                      market_multiplier * 
                      relationship_multiplier * 
                      competitive_multiplier * 
                      technology_multiplier * 
                      custom_multiplier)
        
        # Calculate margin
        total_cost = sum(pricing_data.cost_structure.values())
        margin = (final_price - total_cost) / final_price if final_price > 0 else 0
        
        # Ensure margin stays within acceptable range
        min_margin, max_margin = pricing_data.margin_range
        if margin < min_margin:
            final_price = total_cost / (1 - min_margin)
            margin = min_margin
        elif margin > max_margin:
            final_price = total_cost / (1 - max_margin)
            margin = max_margin
        
        return {
            'final_price': final_price,
            'base_price': base_price,
            'volume_discount': 1 - volume_multiplier,
            'contract_adjustment': contract_multiplier - 1,
            'market_adjustment': market_multiplier - 1,
            'relationship_adjustment': relationship_multiplier - 1,
            'competitive_adjustment': competitive_multiplier - 1,
            'technology_premium': technology_multiplier - 1,
            'custom_adjustment': custom_multiplier - 1,
            'margin_percentage': margin,
            'total_cost': total_cost,
            'total_revenue': final_price * volume
        }
    
    def _calculate_volume_discount(self, volume: int, volume_tiers: Dict[int, float]) -> float:
        """Calculate volume discount multiplier."""
        applicable_discount = 1.0
        
        for tier_volume, discount_multiplier in sorted(volume_tiers.items()):
            if volume >= tier_volume:
                applicable_discount = discount_multiplier
            else:
                break
        
        return applicable_discount
    
    def _calculate_market_adjustment(self, market_condition: MarketCondition, supply_demand_ratio: float) -> float:
        """Calculate market condition pricing adjustment."""
        # Base adjustment from market condition
        condition_adjustments = {
            MarketCondition.CRISIS: 1.5,      # 50% premium in crisis
            MarketCondition.SHORTAGE: 1.3,    # 30% premium in shortage
            MarketCondition.TIGHT: 1.1,       # 10% premium when tight
            MarketCondition.BALANCED: 1.0,    # No adjustment when balanced
            MarketCondition.OVERSUPPLY: 0.9   # 10% discount in oversupply
        }
        
        base_adjustment = condition_adjustments.get(market_condition, 1.0)
        
        # Fine-tune based on exact supply-demand ratio
        if supply_demand_ratio < 0.8:  # Severe shortage
            ratio_adjustment = 1.4
        elif supply_demand_ratio < 1.0:  # Shortage
            ratio_adjustment = 1.0 + (1.0 - supply_demand_ratio) * 0.5
        elif supply_demand_ratio < 1.2:  # Balanced
            ratio_adjustment = 1.0
        elif supply_demand_ratio < 1.5:  # Moderate oversupply
            ratio_adjustment = 1.0 - (supply_demand_ratio - 1.0) * 0.2
        else:  # Significant oversupply
            ratio_adjustment = 0.8
        
        # Blend condition and ratio adjustments
        return (base_adjustment + ratio_adjustment) / 2
    
    def _calculate_technology_premium(self, process_node: str) -> float:
        """Calculate technology premium for advanced process nodes."""
        # Technology premiums based on node advancement
        node_premiums = {
            "3nm": 1.2,    # 20% premium for cutting-edge
            "5nm": 1.15,   # 15% premium
            "7nm": 1.1,    # 10% premium
            "10nm": 1.05,  # 5% premium
            "16nm": 1.0,   # Baseline
            "28nm": 0.98,  # Slight discount
            "65nm": 0.95,  # Mature node discount
            "180nm": 0.90  # Legacy node discount
        }
        
        return node_premiums.get(process_node, 1.0)
    
    def update_market_conditions(self, supply_data: Dict[str, float], demand_data: Dict[str, float]):
        """Update market conditions based on supply and demand data."""
        for process_node in self.supply_demand_ratios.keys():
            supply = supply_data.get(process_node, 0)
            demand = demand_data.get(process_node, 0)
            
            if demand > 0:
                new_ratio = supply / demand
                self.supply_demand_ratios[process_node] = new_ratio
                
                # Update market condition classification
                if new_ratio < 0.7:
                    self.current_market_conditions[process_node] = MarketCondition.CRISIS
                elif new_ratio < 0.9:
                    self.current_market_conditions[process_node] = MarketCondition.SHORTAGE
                elif new_ratio < 1.1:
                    self.current_market_conditions[process_node] = MarketCondition.TIGHT
                elif new_ratio < 1.3:
                    self.current_market_conditions[process_node] = MarketCondition.BALANCED
                else:
                    self.current_market_conditions[process_node] = MarketCondition.OVERSUPPLY
    
    def simulate_price_evolution(self, time_horizon_quarters: int, 
                                demand_growth_rates: Dict[str, float],
                                supply_growth_rates: Dict[str, float]) -> Dict[str, List[float]]:
        """Simulate price evolution over time given demand and supply growth."""
        price_evolution = {}
        
        for product_key, pricing_data in self.market_pricing.items():
            process_node = pricing_data.process_node
            quarterly_prices = [pricing_data.base_price_per_unit]
            
            current_supply_demand_ratio = self.supply_demand_ratios.get(process_node, 1.0)
            
            for quarter in range(1, time_horizon_quarters):
                # Update supply-demand ratio
                demand_growth = demand_growth_rates.get(process_node, 0.02)  # 2% quarterly default
                supply_growth = supply_growth_rates.get(process_node, 0.015)  # 1.5% quarterly default
                
                current_supply_demand_ratio *= (1 + supply_growth) / (1 + demand_growth)
                
                # Calculate market condition from ratio
                if current_supply_demand_ratio < 0.8:
                    market_condition = MarketCondition.SHORTAGE
                elif current_supply_demand_ratio < 1.2:
                    market_condition = MarketCondition.BALANCED
                else:
                    market_condition = MarketCondition.OVERSUPPLY
                
                # Calculate price adjustment
                market_multiplier = self._calculate_market_adjustment(market_condition, current_supply_demand_ratio)
                
                # Add technology learning curve (cost reduction over time)
                learning_rate = 0.95 if "nm" in process_node and int(process_node.replace("nm", "")) <= 10 else 0.98
                cost_reduction = learning_rate ** (quarter / 4)  # Annual learning curve
                
                # Calculate new price
                previous_price = quarterly_prices[-1]
                new_price = previous_price * market_multiplier * cost_reduction
                
                # Add price volatility
                volatility = np.random.normal(0, 0.05)  # 5% price volatility
                new_price *= (1 + volatility)
                
                quarterly_prices.append(max(0, new_price))
            
            price_evolution[product_key] = quarterly_prices
            
            # Update price history
            self.price_history[product_key].extend(quarterly_prices[1:])
        
        return price_evolution
    
    def calculate_market_value(self, demand_forecast: Dict[str, List[float]], 
                             price_evolution: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate total market value and segment analysis."""
        market_analysis = {}
        
        total_market_value_quarterly = []
        time_horizon = min(len(list(demand_forecast.values())[0]), len(list(price_evolution.values())[0]))
        
        for quarter in range(time_horizon):
            quarterly_value = 0
            
            for product_key in self.market_pricing.keys():
                if product_key in demand_forecast and product_key in price_evolution:
                    demand = demand_forecast[product_key][quarter] if quarter < len(demand_forecast[product_key]) else 0
                    price = price_evolution[product_key][quarter] if quarter < len(price_evolution[product_key]) else 0
                    
                    segment_value = demand * price
                    quarterly_value += segment_value
            
            total_market_value_quarterly.append(quarterly_value)
        
        # Calculate segment analysis
        segment_analysis = {}
        for product_key, pricing_data in self.market_pricing.items():
            if product_key in demand_forecast and product_key in price_evolution:
                segment_values = []
                
                for quarter in range(time_horizon):
                    demand = demand_forecast[product_key][quarter] if quarter < len(demand_forecast[product_key]) else 0
                    price = price_evolution[product_key][quarter] if quarter < len(price_evolution[product_key]) else 0
                    segment_values.append(demand * price)
                
                # Calculate metrics
                initial_value = segment_values[0] if segment_values else 0
                final_value = segment_values[-1] if segment_values else 0
                total_value = sum(segment_values)
                
                segment_analysis[product_key] = {
                    'initial_value_billions': initial_value,
                    'final_value_billions': final_value,
                    'total_value_billions': total_value,
                    'value_cagr': (final_value / initial_value) ** (4 / time_horizon) - 1 if initial_value > 0 else 0,
                    'pricing_mechanism': pricing_data.pricing_mechanism.value,
                    'avg_margin': np.mean([self._calculate_current_margin(product_key, quarter) 
                                         for quarter in range(min(4, time_horizon))])
                }
        
        market_analysis = {
            'total_market_value_quarterly': total_market_value_quarterly,
            'initial_market_value': total_market_value_quarterly[0] if total_market_value_quarterly else 0,
            'final_market_value': total_market_value_quarterly[-1] if total_market_value_quarterly else 0,
            'total_cumulative_value': sum(total_market_value_quarterly),
            'market_cagr': ((total_market_value_quarterly[-1] / total_market_value_quarterly[0]) ** (4 / time_horizon) - 1 
                           if total_market_value_quarterly and total_market_value_quarterly[0] > 0 else 0),
            'segment_analysis': segment_analysis,
            'time_horizon_quarters': time_horizon
        }
        
        return market_analysis
    
    def _calculate_current_margin(self, product_key: str, quarter: int = 0) -> float:
        """Calculate current margin for a product."""
        if product_key not in self.market_pricing:
            return 0.0
        
        pricing_data = self.market_pricing[product_key]
        current_price = (self.price_history[product_key][quarter] 
                        if quarter < len(self.price_history[product_key]) 
                        else pricing_data.base_price_per_unit)
        
        total_cost = sum(pricing_data.cost_structure.values())
        margin = (current_price - total_cost) / current_price if current_price > 0 else 0
        
        return max(0, margin)
    
    def get_pricing_summary(self) -> Dict[str, Any]:
        """Get comprehensive pricing model summary."""
        # Calculate average pricing by category
        category_pricing = {}
        for product_key, pricing_data in self.market_pricing.items():
            category = pricing_data.product_category
            if category not in category_pricing:
                category_pricing[category] = []
            category_pricing[category].append(pricing_data.base_price_per_unit)
        
        category_averages = {cat: np.mean(prices) for cat, prices in category_pricing.items()}
        
        # Market condition summary
        condition_distribution = {}
        for condition in self.current_market_conditions.values():
            condition_distribution[condition.value] = condition_distribution.get(condition.value, 0) + 1
        
        return {
            'total_product_categories': len(self.market_pricing),
            'category_avg_pricing': category_averages,
            'market_condition_distribution': condition_distribution,
            'avg_supply_demand_ratio': np.mean(list(self.supply_demand_ratios.values())),
            'avg_competitive_intensity': np.mean(list(self.competitive_intensity.values())),
            'price_range_min': min(p.base_price_per_unit for p in self.market_pricing.values()),
            'price_range_max': max(p.base_price_per_unit for p in self.market_pricing.values()),
            'pricing_mechanisms': {mech.value: sum(1 for p in self.market_pricing.values() 
                                                 if p.pricing_mechanism == mech) 
                                 for mech in PricingMechanism}
        } 