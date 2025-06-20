"""
Company profile generation for ChipGeopolitics simulation.
Creates realistic synthetic data for hyperscalers, chip manufacturers, equipment suppliers, and nation-states.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from config.constants import *


class CompanySize(Enum):
    """Company size categories."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    MEGA = "mega"


class Region(Enum):
    """Geographic regions."""
    NORTH_AMERICA = "North America"
    EUROPE = "Europe"
    ASIA_PACIFIC = "Asia Pacific"
    CHINA = "China"
    TAIWAN = "Taiwan"
    SOUTH_KOREA = "South Korea"
    JAPAN = "Japan"
    GLOBAL = "Global"


@dataclass
class HyperscalerProfile:
    """Profile for hyperscaler companies (cloud/data center operators)."""
    name: str
    size: CompanySize
    annual_capex: float  # Billion USD
    data_center_count: int
    global_regions: List[str]
    primary_region: str
    chip_procurement_volume: int  # Annual accelerators
    cloud_market_share: float  # 0-1
    ai_focus_level: float  # 0-1 (how AI-focused)
    risk_tolerance: float  # 0-1
    geopolitical_stance: float  # 0-1 (0=isolationist, 1=globalist)
    sustainability_commitment: float  # 0-1
    
    # Financial metrics
    annual_revenue: float  # Billion USD
    profit_margin: float  # 0-1
    debt_to_equity: float
    cash_reserves: float  # Billion USD
    
    # Strategic attributes
    vertical_integration_level: float  # 0-1
    innovation_investment_ratio: float  # R&D as % of revenue
    strategic_partnerships: List[str]
    competitive_advantages: List[str]


@dataclass
class ChipManufacturerProfile:
    """Profile for semiconductor manufacturing companies."""
    name: str
    size: CompanySize
    primary_region: str
    foundry_type: str  # "pure-play", "integrated", "specialized"
    
    # Manufacturing capabilities
    process_nodes: List[str]  # e.g., ["28nm", "16nm", "5nm"]
    monthly_capacity: int  # Wafers per month
    capacity_utilization_target: float  # 0-1
    fab_locations: List[str]
    
    # Market position
    market_share: float  # 0-1
    customer_concentration: float  # Herfindahl index
    technology_leadership_score: float  # 0-1
    
    # Financial metrics
    annual_revenue: float  # Billion USD
    capex_intensity: float  # CAPEX as % of revenue
    gross_margin: float  # 0-1
    
    # Strategic attributes
    r_and_d_intensity: float  # R&D as % of revenue
    government_support_level: float  # 0-1
    export_dependency: float  # 0-1
    geopolitical_risk_exposure: float  # 0-1
    
    # Operational metrics
    yield_rates: Dict[str, float]  # Process node -> yield rate
    technology_roadmap_aggressiveness: float  # 0-1
    supply_chain_diversification: float  # 0-1


@dataclass
class EquipmentSupplierProfile:
    """Profile for semiconductor equipment suppliers."""
    name: str
    size: CompanySize
    primary_region: str
    equipment_categories: List[str]  # e.g., ["lithography", "etching", "deposition"]
    
    # Market position
    market_share_by_category: Dict[str, float]
    customer_base_diversity: float  # 0-1
    technology_moat_strength: float  # 0-1
    
    # Financial metrics
    annual_revenue: float  # Billion USD
    operating_margin: float  # 0-1
    order_backlog_months: float
    
    # Strategic attributes
    r_and_d_intensity: float
    patent_portfolio_strength: float  # 0-1
    manufacturing_complexity: float  # 0-1
    export_control_exposure: float  # 0-1
    
    # Operational metrics
    delivery_reliability: float  # 0-1
    service_capability: float  # 0-1
    technology_roadmap_alignment: float  # 0-1


@dataclass
class NationStateProfile:
    """Profile for nation-state actors."""
    name: str
    region: str
    gdp: float  # Trillion USD
    population: int  # Millions
    
    # Semiconductor ecosystem
    domestic_chip_production: float  # Billion USD annually
    chip_self_sufficiency_ratio: float  # 0-1
    semiconductor_companies_count: int
    
    # Policy instruments
    industrial_policy_strength: float  # 0-1
    export_control_capability: float  # 0-1
    subsidy_budget: float  # Billion USD annually
    strategic_stockpile_level: float  # Months of consumption
    
    # Geopolitical attributes
    alliance_network_strength: float  # 0-1
    economic_warfare_propensity: float  # 0-1
    technological_sovereignty_priority: float  # 0-1
    
    # Economic indicators
    tech_sector_gdp_share: float  # 0-1
    innovation_index: float  # 0-1
    supply_chain_resilience_index: float  # 0-1
    
    # Energy and infrastructure
    energy_security_index: float  # 0-1
    renewable_energy_share: float  # 0-1
    grid_stability_index: float  # 0-1


class CompanyProfileGenerator:
    """
    Generates realistic synthetic company profiles based on industry data.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the profile generator."""
        self.random = np.random.RandomState(random_seed)
        
        # Real company templates for realistic generation
        self.hyperscaler_templates = self._load_hyperscaler_templates()
        self.chip_manufacturer_templates = self._load_chip_manufacturer_templates()
        self.equipment_supplier_templates = self._load_equipment_supplier_templates()
        self.nation_state_templates = self._load_nation_state_templates()
    
    def generate_hyperscaler_profiles(self, count: int) -> List[HyperscalerProfile]:
        """Generate hyperscaler company profiles."""
        profiles = []
        
        # Size distribution: 20% mega, 30% large, 30% medium, 20% small
        size_weights = [0.2, 0.3, 0.3, 0.2]
        sizes = self.random.choice(
            [CompanySize.MEGA, CompanySize.LARGE, CompanySize.MEDIUM, CompanySize.SMALL],
            size=count, p=size_weights
        )
        
        for i, size in enumerate(sizes):
            profile = self._generate_single_hyperscaler(f"Hyperscaler_{i+1}", size)
            profiles.append(profile)
        
        return profiles
    
    def _generate_single_hyperscaler(self, name: str, size: CompanySize) -> HyperscalerProfile:
        """Generate a single hyperscaler profile."""
        # Size-based parameters
        if size == CompanySize.MEGA:
            capex_range = (50, 100)  # Billion USD
            revenue_range = (200, 500)
            dc_count_range = (200, 500)
            chip_volume_range = (500000, 2000000)
        elif size == CompanySize.LARGE:
            capex_range = (15, 50)
            revenue_range = (50, 200)
            dc_count_range = (50, 200)
            chip_volume_range = (100000, 500000)
        elif size == CompanySize.MEDIUM:
            capex_range = (5, 15)
            revenue_range = (10, 50)
            dc_count_range = (10, 50)
            chip_volume_range = (10000, 100000)
        else:  # SMALL
            capex_range = (1, 5)
            revenue_range = (1, 10)
            dc_count_range = (5, 20)
            chip_volume_range = (1000, 10000)
        
        # Regional distribution
        regions = list(Region)
        primary_region = self.random.choice([
            Region.NORTH_AMERICA, Region.EUROPE, Region.ASIA_PACIFIC, Region.CHINA
        ]).value
        
        # Global presence based on size
        global_presence_count = {
            CompanySize.MEGA: self.random.randint(5, 8),
            CompanySize.LARGE: self.random.randint(3, 6),
            CompanySize.MEDIUM: self.random.randint(2, 4),
            CompanySize.SMALL: self.random.randint(1, 3)
        }[size]
        
        global_regions = self.random.choice(
            [r.value for r in regions if r.value != "Global"],
            size=min(global_presence_count, len(regions)-1),
            replace=False
        ).tolist()
        
        return HyperscalerProfile(
            name=name,
            size=size,
            annual_capex=self.random.uniform(*capex_range),
            data_center_count=self.random.randint(*dc_count_range),
            global_regions=global_regions,
            primary_region=primary_region,
            chip_procurement_volume=self.random.randint(*chip_volume_range),
            cloud_market_share=self._generate_market_share(size),
            ai_focus_level=self.random.beta(2, 2),  # Concentrated around 0.5
            risk_tolerance=self.random.beta(2, 3),  # Slightly risk-averse
            geopolitical_stance=self.random.beta(3, 2),  # Slightly globalist
            sustainability_commitment=self.random.beta(3, 2),  # High commitment trend
            
            # Financial metrics
            annual_revenue=self.random.uniform(*revenue_range),
            profit_margin=self.random.uniform(0.1, 0.3),
            debt_to_equity=self.random.uniform(0.1, 0.8),
            cash_reserves=self.random.uniform(capex_range[0]/2, capex_range[1]*2),
            
            # Strategic attributes
            vertical_integration_level=self.random.beta(2, 3),
            innovation_investment_ratio=self.random.uniform(0.08, 0.20),
            strategic_partnerships=self._generate_partnerships("hyperscaler"),
            competitive_advantages=self._generate_competitive_advantages("hyperscaler")
        )
    
    def generate_chip_manufacturer_profiles(self, count: int) -> List[ChipManufacturerProfile]:
        """Generate chip manufacturer company profiles."""
        profiles = []
        
        # Size distribution based on industry structure
        size_weights = [0.1, 0.2, 0.4, 0.3]  # Few mega, some large, many medium/small
        sizes = self.random.choice(
            [CompanySize.MEGA, CompanySize.LARGE, CompanySize.MEDIUM, CompanySize.SMALL],
            size=count, p=size_weights
        )
        
        for i, size in enumerate(sizes):
            profile = self._generate_single_chip_manufacturer(f"ChipMfg_{i+1}", size)
            profiles.append(profile)
        
        return profiles
    
    def _generate_single_chip_manufacturer(self, name: str, size: CompanySize) -> ChipManufacturerProfile:
        """Generate a single chip manufacturer profile."""
        # Foundry type distribution
        foundry_types = ["pure-play", "integrated", "specialized"]
        foundry_weights = [0.4, 0.4, 0.2]
        foundry_type = self.random.choice(foundry_types, p=foundry_weights)
        
        # Process node capabilities based on size and type
        all_nodes = ["180nm", "90nm", "65nm", "40nm", "28nm", "16nm", "10nm", "7nm", "5nm", "3nm", "2nm"]
        
        if size == CompanySize.MEGA:
            # Leading edge capabilities
            node_count = self.random.randint(6, 10)
            available_nodes = all_nodes[-node_count:]  # Latest nodes
            revenue_range = (30, 80)
            capacity_range = (100000, 500000)  # Wafers per month
        elif size == CompanySize.LARGE:
            node_count = self.random.randint(4, 8)
            start_idx = self.random.randint(0, 3)
            available_nodes = all_nodes[start_idx:start_idx+node_count]
            revenue_range = (10, 30)
            capacity_range = (50000, 150000)
        elif size == CompanySize.MEDIUM:
            node_count = self.random.randint(3, 6)
            start_idx = self.random.randint(0, 5)
            available_nodes = all_nodes[start_idx:start_idx+node_count]
            revenue_range = (2, 10)
            capacity_range = (10000, 50000)
        else:  # SMALL
            node_count = self.random.randint(2, 4)
            start_idx = self.random.randint(0, 7)
            available_nodes = all_nodes[start_idx:start_idx+node_count]
            revenue_range = (0.5, 2)
            capacity_range = (1000, 10000)
        
        # Regional distribution (Asia-heavy for semiconductors)
        region_weights = {
            Region.TAIWAN.value: 0.25,
            Region.SOUTH_KOREA.value: 0.20,
            Region.CHINA.value: 0.20,
            Region.JAPAN.value: 0.15,
            Region.NORTH_AMERICA.value: 0.10,
            Region.EUROPE.value: 0.10
        }
        primary_region = self.random.choice(
            list(region_weights.keys()),
            p=list(region_weights.values())
        )
        
        # Generate yield rates for each process node
        yield_rates = {}
        for node in available_nodes:
            # More advanced nodes have lower yields
            node_complexity = all_nodes.index(node) / len(all_nodes)
            base_yield = 0.95 - (node_complexity * 0.3)  # 95% for old, 65% for newest
            yield_rates[node] = max(0.3, self.random.normal(base_yield, 0.05))
        
        return ChipManufacturerProfile(
            name=name,
            size=size,
            primary_region=primary_region,
            foundry_type=foundry_type,
            process_nodes=available_nodes,
            monthly_capacity=self.random.randint(*capacity_range),
            capacity_utilization_target=self.random.uniform(0.80, 0.95),
            fab_locations=self._generate_fab_locations(size, primary_region),
            market_share=self._generate_market_share(size),
            customer_concentration=self.random.uniform(0.2, 0.8),
            technology_leadership_score=self._calculate_tech_leadership(available_nodes, all_nodes),
            annual_revenue=self.random.uniform(*revenue_range),
            capex_intensity=self.random.uniform(0.15, 0.35),
            gross_margin=self.random.uniform(0.25, 0.55),
            r_and_d_intensity=self.random.uniform(0.08, 0.25),
            government_support_level=self._calculate_government_support(primary_region),
            export_dependency=self.random.uniform(0.3, 0.9),
            geopolitical_risk_exposure=self._calculate_geopolitical_risk(primary_region),
            yield_rates=yield_rates,
            technology_roadmap_aggressiveness=self.random.beta(2, 2),
            supply_chain_diversification=self.random.beta(2, 3)
        )
    
    def generate_equipment_supplier_profiles(self, count: int) -> List[EquipmentSupplierProfile]:
        """Generate equipment supplier company profiles."""
        profiles = []
        
        # Equipment categories
        equipment_categories = [
            "lithography", "etching", "deposition", "ion_implantation",
            "chemical_mechanical_planarization", "inspection", "metrology",
            "packaging", "assembly", "test"
        ]
        
        size_weights = [0.05, 0.15, 0.40, 0.40]  # Very few mega, concentrated industry
        sizes = self.random.choice(
            [CompanySize.MEGA, CompanySize.LARGE, CompanySize.MEDIUM, CompanySize.SMALL],
            size=count, p=size_weights
        )
        
        for i, size in enumerate(sizes):
            profile = self._generate_single_equipment_supplier(f"Equipment_{i+1}", size, equipment_categories)
            profiles.append(profile)
        
        return profiles
    
    def _generate_single_equipment_supplier(self, name: str, size: CompanySize, equipment_categories: List[str]) -> EquipmentSupplierProfile:
        """Generate a single equipment supplier profile."""
        # Regional distribution (developed markets for high-tech equipment)
        region_weights = {
            Region.NORTH_AMERICA.value: 0.30,
            Region.EUROPE.value: 0.25,
            Region.JAPAN.value: 0.25,
            Region.SOUTH_KOREA.value: 0.10,
            Region.TAIWAN.value: 0.05,
            Region.CHINA.value: 0.05
        }
        primary_region = self.random.choice(
            list(region_weights.keys()),
            p=list(region_weights.values())
        )
        
        # Equipment specialization based on size
        if size == CompanySize.MEGA:
            num_categories = self.random.randint(4, 8)
            revenue_range = (10, 25)
        elif size == CompanySize.LARGE:
            num_categories = self.random.randint(2, 5)
            revenue_range = (3, 10)
        elif size == CompanySize.MEDIUM:
            num_categories = self.random.randint(1, 3)
            revenue_range = (0.5, 3)
        else:  # SMALL
            num_categories = self.random.randint(1, 2)
            revenue_range = (0.1, 0.5)
        
        company_categories = self.random.choice(
            equipment_categories,
            size=min(num_categories, len(equipment_categories)),
            replace=False
        ).tolist()
        
        # Generate market share by category
        market_share_by_category = {}
        for category in company_categories:
            if size == CompanySize.MEGA and category in ["lithography", "etching"]:
                # Dominant players in critical categories
                share = self.random.uniform(0.30, 0.70)
            else:
                share = self._generate_market_share(size)
            market_share_by_category[category] = share
        
        return EquipmentSupplierProfile(
            name=name,
            size=size,
            primary_region=primary_region,
            equipment_categories=company_categories,
            market_share_by_category=market_share_by_category,
            customer_base_diversity=self.random.beta(3, 2),
            technology_moat_strength=self._calculate_tech_moat(company_categories),
            annual_revenue=self.random.uniform(*revenue_range),
            operating_margin=self.random.uniform(0.15, 0.35),
            order_backlog_months=self.random.uniform(6, 24),
            r_and_d_intensity=self.random.uniform(0.12, 0.25),
            patent_portfolio_strength=self.random.beta(3, 2),
            manufacturing_complexity=self._calculate_manufacturing_complexity(company_categories),
            export_control_exposure=self._calculate_export_control_exposure(primary_region, company_categories),
            delivery_reliability=self.random.beta(4, 2),
            service_capability=self.random.beta(3, 2),
            technology_roadmap_alignment=self.random.beta(3, 2)
        )
    
    def generate_nation_state_profiles(self, count: int) -> List[NationStateProfile]:
        """Generate nation-state profiles."""
        profiles = []
        
        # Major semiconductor-relevant nations
        nation_templates = [
            ("United States", Region.NORTH_AMERICA.value),
            ("China", Region.CHINA.value),
            ("Taiwan", Region.TAIWAN.value),
            ("South Korea", Region.SOUTH_KOREA.value),
            ("Japan", Region.JAPAN.value),
            ("Germany", Region.EUROPE.value),
            ("Netherlands", Region.EUROPE.value),
            ("Singapore", Region.ASIA_PACIFIC.value),
            ("Israel", Region.ASIA_PACIFIC.value),
            ("India", Region.ASIA_PACIFIC.value)
        ]
        
        # Add more synthetic nations if needed
        for i in range(max(0, count - len(nation_templates))):
            region_choice = self.random.choice(list(Region)[:7])  # Exclude GLOBAL
            nation_templates.append((f"Nation_{i+1}", region_choice.value))
        
        # Select requested number of nations
        available_nations = nation_templates[:min(count, len(nation_templates))]
        if len(available_nations) < count:
            # Add more nations if needed
            for i in range(count - len(available_nations)):
                region_choice = self.random.choice([r for r in Region if r.value != "Global"])
                available_nations.append((f"Nation_{len(nation_templates)+i+1}", region_choice.value))
        
        selected_nations = available_nations
        
        for name, region in selected_nations:
            profile = self._generate_single_nation_state(name, region)
            profiles.append(profile)
        
        return profiles
    
    def _generate_single_nation_state(self, name: str, region: str) -> NationStateProfile:
        """Generate a single nation-state profile."""
        # GDP and population based on region and real-world data
        if region == Region.NORTH_AMERICA.value and "United States" in name:
            gdp = self.random.uniform(22, 26)  # Trillion USD
            population = self.random.randint(320, 340)  # Millions
            chip_production = self.random.uniform(40, 60)  # Billion USD
            self_sufficiency = self.random.uniform(0.15, 0.25)
        elif region == Region.CHINA.value:
            gdp = self.random.uniform(16, 20)
            population = self.random.randint(1400, 1450)
            chip_production = self.random.uniform(25, 35)
            self_sufficiency = self.random.uniform(0.20, 0.30)
        elif region == Region.TAIWAN.value:
            gdp = self.random.uniform(0.7, 0.8)
            population = self.random.randint(23, 24)
            chip_production = self.random.uniform(150, 200)  # Very high per capita
            self_sufficiency = self.random.uniform(0.95, 1.0)  # Nearly self-sufficient in production
        elif region == Region.SOUTH_KOREA.value:
            gdp = self.random.uniform(1.8, 2.2)
            population = self.random.randint(51, 52)
            chip_production = self.random.uniform(80, 120)
            self_sufficiency = self.random.uniform(0.70, 0.90)
        elif region == Region.JAPAN.value:
            gdp = self.random.uniform(4.5, 5.5)
            population = self.random.randint(125, 127)
            chip_production = self.random.uniform(30, 50)
            self_sufficiency = self.random.uniform(0.40, 0.60)
        else:  # Other regions
            gdp = self.random.uniform(0.5, 5.0)
            population = self.random.randint(5, 100)
            chip_production = self.random.uniform(1, 20)
            self_sufficiency = self.random.uniform(0.05, 0.40)
        
        return NationStateProfile(
            name=name,
            region=region,
            gdp=gdp,
            population=population,
            domestic_chip_production=chip_production,
            chip_self_sufficiency_ratio=self_sufficiency,
            semiconductor_companies_count=self.random.randint(5, 100),
            industrial_policy_strength=self._calculate_industrial_policy_strength(region),
            export_control_capability=self._calculate_export_control_capability(region),
            subsidy_budget=self.random.uniform(0.5, 50),
            strategic_stockpile_level=self.random.uniform(1, 12),  # Months
            alliance_network_strength=self._calculate_alliance_strength(region, name),
            economic_warfare_propensity=self.random.beta(2, 4),  # Most nations prefer cooperation
            technological_sovereignty_priority=self._calculate_sovereignty_priority(region),
            tech_sector_gdp_share=self.random.uniform(0.05, 0.25),
            innovation_index=self.random.beta(3, 2),
            supply_chain_resilience_index=self.random.beta(2, 3),
            energy_security_index=self.random.beta(2, 2),
            renewable_energy_share=self.random.uniform(0.10, 0.80),
            grid_stability_index=self.random.beta(3, 2)
        )
    
    # Helper methods for profile generation
    def _generate_market_share(self, size: CompanySize) -> float:
        """Generate realistic market share based on company size."""
        if size == CompanySize.MEGA:
            return self.random.uniform(0.15, 0.40)
        elif size == CompanySize.LARGE:
            return self.random.uniform(0.05, 0.15)
        elif size == CompanySize.MEDIUM:
            return self.random.uniform(0.01, 0.05)
        else:  # SMALL
            return self.random.uniform(0.001, 0.01)
    
    def _generate_partnerships(self, agent_type: str) -> List[str]:
        """Generate strategic partnerships."""
        partnership_types = {
            "hyperscaler": ["cloud_alliance", "ai_research", "sustainability", "supply_chain"],
            "chip_manufacturer": ["foundry_services", "technology_development", "capacity_sharing"],
            "equipment_supplier": ["joint_development", "service_partnership", "technology_licensing"]
        }
        
        available_types = partnership_types.get(agent_type, [])
        num_partnerships = self.random.randint(1, len(available_types))
        return self.random.choice(available_types, size=num_partnerships, replace=False).tolist()
    
    def _generate_competitive_advantages(self, agent_type: str) -> List[str]:
        """Generate competitive advantages."""
        advantages = {
            "hyperscaler": ["scale_economics", "global_presence", "ai_expertise", "ecosystem_integration"],
            "chip_manufacturer": ["process_technology", "manufacturing_efficiency", "customer_relationships"],
            "equipment_supplier": ["technology_leadership", "service_excellence", "patent_portfolio"]
        }
        
        available_advantages = advantages.get(agent_type, [])
        num_advantages = self.random.randint(1, min(3, len(available_advantages)))
        return self.random.choice(available_advantages, size=num_advantages, replace=False).tolist()
    
    def _generate_fab_locations(self, size: CompanySize, primary_region: str) -> List[str]:
        """Generate fab locations based on size and primary region."""
        all_regions = [r.value for r in Region if r.value != "Global"]
        
        if size == CompanySize.MEGA:
            num_locations = self.random.randint(3, 6)
        elif size == CompanySize.LARGE:
            num_locations = self.random.randint(2, 4)
        else:
            num_locations = self.random.randint(1, 2)
        
        # Always include primary region
        locations = [primary_region]
        
        # Add additional regions
        remaining_regions = [r for r in all_regions if r != primary_region]
        additional_count = min(num_locations - 1, len(remaining_regions))
        
        if additional_count > 0:
            additional_locations = self.random.choice(
                remaining_regions, size=additional_count, replace=False
            ).tolist()
            locations.extend(additional_locations)
        
        return locations
    
    def _calculate_tech_leadership(self, available_nodes: List[str], all_nodes: List[str]) -> float:
        """Calculate technology leadership score based on process node capabilities."""
        if not available_nodes:
            return 0.0
        
        # Score based on most advanced node and breadth of capabilities
        most_advanced_idx = max(all_nodes.index(node) for node in available_nodes)
        breadth_score = len(available_nodes) / len(all_nodes)
        advancement_score = most_advanced_idx / len(all_nodes)
        
        return (advancement_score * 0.7 + breadth_score * 0.3)
    
    def _calculate_government_support(self, region: str) -> float:
        """Calculate government support level based on region."""
        support_levels = {
            Region.CHINA.value: 0.9,
            Region.TAIWAN.value: 0.8,
            Region.SOUTH_KOREA.value: 0.7,
            Region.JAPAN.value: 0.6,
            Region.EUROPE.value: 0.5,
            Region.NORTH_AMERICA.value: 0.4
        }
        base_level = support_levels.get(region, 0.3)
        return min(1.0, self.random.normal(base_level, 0.1))
    
    def _calculate_geopolitical_risk(self, region: str) -> float:
        """Calculate geopolitical risk exposure based on region."""
        risk_levels = {
            Region.CHINA.value: 0.8,
            Region.TAIWAN.value: 0.9,  # High due to tensions
            Region.SOUTH_KOREA.value: 0.6,
            Region.JAPAN.value: 0.4,
            Region.EUROPE.value: 0.3,
            Region.NORTH_AMERICA.value: 0.2
        }
        base_risk = risk_levels.get(region, 0.5)
        return max(0.0, min(1.0, self.random.normal(base_risk, 0.1)))
    
    def _calculate_tech_moat(self, categories: List[str]) -> float:
        """Calculate technology moat strength for equipment suppliers."""
        critical_categories = ["lithography", "etching", "ion_implantation"]
        moat_strength = 0.3  # Base level
        
        for category in categories:
            if category in critical_categories:
                moat_strength += 0.2
            else:
                moat_strength += 0.1
        
        return min(1.0, moat_strength)
    
    def _calculate_manufacturing_complexity(self, categories: List[str]) -> float:
        """Calculate manufacturing complexity for equipment suppliers."""
        complexity_scores = {
            "lithography": 1.0,
            "ion_implantation": 0.9,
            "etching": 0.8,
            "deposition": 0.7,
            "inspection": 0.6,
            "metrology": 0.6,
            "packaging": 0.4,
            "assembly": 0.3,
            "test": 0.3
        }
        
        if not categories:
            return 0.5
        
        total_complexity = sum(complexity_scores.get(cat, 0.5) for cat in categories)
        return min(1.0, total_complexity / len(categories))
    
    def _calculate_export_control_exposure(self, region: str, categories: List[str]) -> float:
        """Calculate export control exposure for equipment suppliers."""
        # Higher exposure for non-allied regions and critical technologies
        base_exposure = {
            Region.CHINA.value: 0.9,
            Region.NORTH_AMERICA.value: 0.1,
            Region.EUROPE.value: 0.2,
            Region.JAPAN.value: 0.2,
            Region.SOUTH_KOREA.value: 0.3,
            Region.TAIWAN.value: 0.4
        }.get(region, 0.5)
        
        critical_tech_exposure = 0.0
        critical_categories = ["lithography", "etching", "ion_implantation"]
        
        for category in categories:
            if category in critical_categories:
                critical_tech_exposure += 0.2
        
        return min(1.0, base_exposure + critical_tech_exposure)
    
    def _calculate_industrial_policy_strength(self, region: str) -> float:
        """Calculate industrial policy strength for nation-states."""
        policy_strengths = {
            Region.CHINA.value: 0.95,
            Region.SOUTH_KOREA.value: 0.85,
            Region.TAIWAN.value: 0.80,
            Region.JAPAN.value: 0.70,
            Region.EUROPE.value: 0.60,
            Region.NORTH_AMERICA.value: 0.50
        }
        base_strength = policy_strengths.get(region, 0.40)
        return max(0.0, min(1.0, self.random.normal(base_strength, 0.1)))
    
    def _calculate_export_control_capability(self, region: str) -> float:
        """Calculate export control capability for nation-states."""
        capabilities = {
            Region.NORTH_AMERICA.value: 0.95,
            Region.EUROPE.value: 0.80,
            Region.JAPAN.value: 0.70,
            Region.SOUTH_KOREA.value: 0.60,
            Region.TAIWAN.value: 0.40,
            Region.CHINA.value: 0.30
        }
        base_capability = capabilities.get(region, 0.20)
        return max(0.0, min(1.0, self.random.normal(base_capability, 0.1)))
    
    def _calculate_alliance_strength(self, region: str, name: str) -> float:
        """Calculate alliance network strength for nation-states."""
        # US and allies have strong networks
        if region == Region.NORTH_AMERICA.value or "United States" in name:
            return self.random.uniform(0.85, 0.95)
        elif region in [Region.EUROPE.value, Region.JAPAN.value, Region.SOUTH_KOREA.value, Region.TAIWAN.value]:
            return self.random.uniform(0.70, 0.90)
        elif region == Region.CHINA.value:
            return self.random.uniform(0.30, 0.50)
        else:
            return self.random.uniform(0.20, 0.70)
    
    def _calculate_sovereignty_priority(self, region: str) -> float:
        """Calculate technological sovereignty priority for nation-states."""
        priorities = {
            Region.CHINA.value: 0.95,
            Region.EUROPE.value: 0.75,
            Region.JAPAN.value: 0.65,
            Region.SOUTH_KOREA.value: 0.70,
            Region.TAIWAN.value: 0.80,
            Region.NORTH_AMERICA.value: 0.60
        }
        base_priority = priorities.get(region, 0.50)
        return max(0.0, min(1.0, self.random.normal(base_priority, 0.1)))
    
    # Template loading methods (would load from files in real implementation)
    def _load_hyperscaler_templates(self) -> Dict[str, Any]:
        """Load hyperscaler company templates."""
        return {
            "major_cloud_providers": ["AWS", "Microsoft Azure", "Google Cloud", "Alibaba Cloud"],
            "data_center_operators": ["Digital Realty", "Equinix", "NTT Communications"],
            "ai_specialists": ["NVIDIA", "Cerebras", "SambaNova"]
        }
    
    def _load_chip_manufacturer_templates(self) -> Dict[str, Any]:
        """Load chip manufacturer templates."""
        return {
            "foundries": ["TSMC", "Samsung", "GlobalFoundries", "SMIC"],
            "memory": ["SK Hynix", "Micron", "Kioxia"],
            "specialized": ["Analog Devices", "Infineon", "STMicroelectronics"]
        }
    
    def _load_equipment_supplier_templates(self) -> Dict[str, Any]:
        """Load equipment supplier templates."""
        return {
            "lithography": ["ASML", "Canon", "Nikon"],
            "etching": ["Applied Materials", "Lam Research", "Tokyo Electron"],
            "deposition": ["Applied Materials", "AMAT", "Veeco"]
        }
    
    def _load_nation_state_templates(self) -> Dict[str, Any]:
        """Load nation-state templates."""
        return {
            "major_powers": ["United States", "China", "European Union"],
            "semiconductor_hubs": ["Taiwan", "South Korea", "Japan", "Singapore"],
            "emerging_players": ["India", "Vietnam", "Malaysia"]
        }
    
    def save_profiles_to_json(self, profiles: List[Any], filename: str, output_dir: Path) -> None:
        """Save profiles to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass instances to dictionaries
        profiles_data = [asdict(profile) for profile in profiles]
        
        with open(output_dir / filename, 'w') as f:
            json.dump(profiles_data, f, indent=2, default=str)
    
    def save_profiles_to_csv(self, profiles: List[Any], filename: str, output_dir: Path) -> None:
        """Save profiles to CSV file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        profiles_data = [asdict(profile) for profile in profiles]
        df = pd.json_normalize(profiles_data)
        
        df.to_csv(output_dir / filename, index=False) 