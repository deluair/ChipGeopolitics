"""
Export Control Simulation for Semiconductor Industry

Models comprehensive export control regimes, technology restrictions, licensing processes,
and compliance impacts on semiconductor supply chains and technology transfer.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import itertools

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class ControlRegime(Enum):
    """Export control regimes."""
    EAR = "export_administration_regulations"  # US EAR
    ITAR = "international_traffic_arms_regulations"  # US ITAR
    EU_DUAL_USE = "eu_dual_use_regulation"  # EU Dual-Use
    WASSENAAR = "wassenaar_arrangement"  # Multilateral
    AUSTRALIA_GROUP = "australia_group"  # Chemical/biological
    MTCR = "missile_technology_control_regime"  # Missile technology
    NSG = "nuclear_suppliers_group"  # Nuclear
    UNILATERAL = "unilateral_controls"  # Country-specific

class ControlLevel(Enum):
    """Control classification levels."""
    UNRESTRICTED = "unrestricted"
    CONTROLLED = "controlled"
    RESTRICTED = "restricted" 
    PROHIBITED = "prohibited"
    CLASSIFIED = "classified"

class LicenseType(Enum):
    """Types of export licenses."""
    GENERAL = "general_license"
    SPECIFIC = "specific_license"
    GLOBAL = "global_license"
    DEEMED_EXPORT = "deemed_export_license"
    TEMPORARY_IMPORT = "temporary_import"
    LICENSE_EXCEPTION = "license_exception"
    NO_LICENSE_REQUIRED = "no_license_required"

class EntityListStatus(Enum):
    """Entity list classifications."""
    CLEARED = "cleared"
    UNVERIFIED = "unverified_list"
    DENIED_PERSONS = "denied_persons_list"
    ENTITY_LIST = "entity_list"
    MILITARY_END_USER = "military_end_user_list"
    SECTORAL_SANCTIONS = "sectoral_sanctions"

@dataclass
class ControlledTechnology:
    """Controlled technology specification."""
    technology_id: str
    name: str
    eccn: str  # Export Control Classification Number
    control_regime: ControlRegime
    control_level: ControlLevel
    controlled_parameters: Dict[str, Any]  # Technical parameters under control
    countries_controlled: List[str]
    end_use_restrictions: List[str]
    license_requirements: Dict[str, LicenseType]
    compliance_cost_factor: float  # Additional cost multiplier

@dataclass
class ExportLicense:
    """Export license instance."""
    license_id: str
    license_type: LicenseType
    technology_id: str
    exporter: str
    importer: str
    end_user: str
    quantity: float
    value_usd: float
    issue_date: datetime
    expiration_date: datetime
    conditions: List[str]
    approval_probability: float
    processing_time_days: int

@dataclass
class ComplianceAssessment:
    """Compliance assessment result."""
    transaction_id: str
    technology_id: str
    source_country: str
    destination_country: str
    end_user_entity: str
    compliance_status: str  # "compliant", "requires_license", "prohibited"
    required_licenses: List[LicenseType]
    entity_list_flags: List[EntityListStatus]
    estimated_delay_days: int
    compliance_cost_usd: float
    risk_factors: List[str]

class ExportControlSimulator:
    """
    Comprehensive export control simulation for semiconductor industry.
    
    Models:
    - Technology classification and control lists
    - License requirement determination
    - Entity list screening
    - Compliance cost calculation
    - Geopolitical restriction evolution
    - Technology transfer limitations
    """
    
    def __init__(self):
        # Control regime data
        self.controlled_technologies: Dict[str, ControlledTechnology] = {}
        self.entity_lists: Dict[str, Dict[str, EntityListStatus]] = {}
        self.country_relationships: Dict[Tuple[str, str], float] = {}  # Bilateral relationship scores
        
        # Simulation state
        self.active_licenses: Dict[str, ExportLicense] = {}
        self.compliance_history: List[ComplianceAssessment] = []
        self.policy_changes: List[Dict[str, Any]] = []
        
        # Economic impact tracking
        self.trade_flow_impacts: Dict[str, float] = {}
        self.technology_diffusion_delays: Dict[str, int] = {}
        
        # Initialize with realistic 2024 export control landscape
        self._initialize_controlled_technologies()
        self._initialize_entity_lists()
        self._initialize_country_relationships()
    
    def _initialize_controlled_technologies(self):
        """Initialize realistic controlled technology catalog."""
        
        # Semiconductor manufacturing equipment (high-end)
        self.controlled_technologies["euv_lithography"] = ControlledTechnology(
            technology_id="euv_lithography",
            name="Extreme Ultraviolet Lithography Systems",
            eccn="3B001",
            control_regime=ControlRegime.WASSENAAR,
            control_level=ControlLevel.RESTRICTED,
            controlled_parameters={
                "wavelength_nm": 13.5,
                "resolution_nm": [5, 10],  # 5nm and below restricted
                "throughput_wph": 200
            },
            countries_controlled=["China", "Russia", "Iran", "North Korea"],
            end_use_restrictions=["military", "nuclear", "missile"],
            license_requirements={
                "China": LicenseType.SPECIFIC,
                "Russia": LicenseType.PROHIBITED,
                "default": LicenseType.GENERAL
            },
            compliance_cost_factor=1.15
        )
        
        self.controlled_technologies["advanced_semiconductors"] = ControlledTechnology(
            technology_id="advanced_semiconductors",
            name="Advanced Semiconductors",
            eccn="3A001",
            control_regime=ControlRegime.EAR,
            control_level=ControlLevel.CONTROLLED,
            controlled_parameters={
                "process_node_nm": [1, 2, 3, 4, 5, 7],  # Advanced nodes
                "logic_performance_tops": 2.0,  # AI performance threshold
                "memory_bandwidth_gbps": 600
            },
            countries_controlled=["China", "Russia"],
            end_use_restrictions=["military", "supercomputing", "ai_training"],
            license_requirements={
                "China": LicenseType.SPECIFIC,
                "Russia": LicenseType.PROHIBITED,
                "default": LicenseType.NO_LICENSE_REQUIRED
            },
            compliance_cost_factor=1.08
        )
        
        self.controlled_technologies["chip_design_software"] = ControlledTechnology(
            technology_id="chip_design_software",
            name="Semiconductor Design Software",
            eccn="5D002",
            control_regime=ControlRegime.EAR,
            control_level=ControlLevel.CONTROLLED,
            controlled_parameters={
                "design_rule_nm": 16,  # 16nm and below
                "eda_capabilities": ["physical_design", "verification", "dft"],
                "ip_libraries": ["advanced_processors", "ai_accelerators"]
            },
            countries_controlled=["China", "Russia", "Iran"],
            end_use_restrictions=["military", "nuclear"],
            license_requirements={
                "China": LicenseType.SPECIFIC,
                "Russia": LicenseType.SPECIFIC,
                "default": LicenseType.GENERAL
            },
            compliance_cost_factor=1.12
        )
        
        self.controlled_technologies["epitaxial_deposition"] = ControlledTechnology(
            technology_id="epitaxial_deposition",
            name="Epitaxial Deposition Equipment",
            eccn="3B001",
            control_regime=ControlRegime.WASSENAAR,
            control_level=ControlLevel.CONTROLLED,
            controlled_parameters={
                "temperature_celsius": 1200,
                "uniformity_percent": 2.0,
                "wafer_size_mm": [200, 300]
            },
            countries_controlled=["China", "Russia"],
            end_use_restrictions=["military"],
            license_requirements={
                "China": LicenseType.SPECIFIC,
                "Russia": LicenseType.SPECIFIC,
                "default": LicenseType.GENERAL
            },
            compliance_cost_factor=1.10
        )
        
        self.controlled_technologies["ion_implantation"] = ControlledTechnology(
            technology_id="ion_implantation",
            name="Ion Implantation Equipment",
            eccn="3B001",
            control_regime=ControlRegime.WASSENAAR,
            control_level=ControlLevel.CONTROLLED,
            controlled_parameters={
                "beam_current_ma": 30,
                "energy_kev": 200,
                "dose_accuracy_percent": 1.0
            },
            countries_controlled=["China", "Russia"],
            end_use_restrictions=["military", "nuclear"],
            license_requirements={
                "China": LicenseType.SPECIFIC,
                "Russia": LicenseType.SPECIFIC,
                "default": LicenseType.GENERAL
            },
            compliance_cost_factor=1.08
        )
        
        # Materials and chemicals
        self.controlled_technologies["rare_earth_elements"] = ControlledTechnology(
            technology_id="rare_earth_elements",
            name="Rare Earth Elements for Semiconductors",
            eccn="1C002",
            control_regime=ControlRegime.EAR,
            control_level=ControlLevel.CONTROLLED,
            controlled_parameters={
                "purity_percent": 99.9,
                "elements": ["yttrium", "europium", "terbium", "dysprosium"],
                "processing_grade": "semiconductor"
            },
            countries_controlled=["Iran", "North Korea"],
            end_use_restrictions=["nuclear", "missile"],
            license_requirements={
                "Iran": LicenseType.PROHIBITED,
                "North Korea": LicenseType.PROHIBITED,
                "default": LicenseType.NO_LICENSE_REQUIRED
            },
            compliance_cost_factor=1.05
        )
    
    def _initialize_entity_lists(self):
        """Initialize entity list classifications by country."""
        
        # US Entity List (as of 2024)
        self.entity_lists["USA"] = {
            # Chinese entities
            "SMIC": EntityListStatus.ENTITY_LIST,
            "Huawei": EntityListStatus.ENTITY_LIST,
            "YMTC": EntityListStatus.ENTITY_LIST,
            "CXMT": EntityListStatus.ENTITY_LIST,
            "Phytium": EntityListStatus.ENTITY_LIST,
            "Cambricon": EntityListStatus.ENTITY_LIST,
            
            # Russian entities
            "Mikron": EntityListStatus.ENTITY_LIST,
            "Angstrem": EntityListStatus.ENTITY_LIST,
            
            # Iranian entities
            "AEOI": EntityListStatus.DENIED_PERSONS,
            
            # Cleared major entities
            "TSMC": EntityListStatus.CLEARED,
            "Samsung": EntityListStatus.CLEARED,
            "SK Hynix": EntityListStatus.CLEARED,
            "Sony": EntityListStatus.CLEARED,
            "ASML": EntityListStatus.CLEARED
        }
        
        # EU restrictions
        self.entity_lists["EU"] = {
            "Huawei": EntityListStatus.RESTRICTED,
            "ZTE": EntityListStatus.RESTRICTED,
            "Russian_Military": EntityListStatus.SECTORAL_SANCTIONS,
            "TSMC": EntityListStatus.CLEARED,
            "Samsung": EntityListStatus.CLEARED
        }
        
        # Japanese controls
        self.entity_lists["Japan"] = {
            "Chinese_Military": EntityListStatus.RESTRICTED,
            "Russian_Entities": EntityListStatus.SECTORAL_SANCTIONS,
            "TSMC": EntityListStatus.CLEARED,
            "Samsung": EntityListStatus.CLEARED
        }
    
    def _initialize_country_relationships(self):
        """Initialize bilateral relationship scores (0-1, higher = better relations)."""
        
        relationships = [
            # US relationships
            ("USA", "Taiwan", 0.9),
            ("USA", "Japan", 0.95),
            ("USA", "South Korea", 0.9),
            ("USA", "Netherlands", 0.95),
            ("USA", "UK", 0.98),
            ("USA", "China", 0.3),
            ("USA", "Russia", 0.1),
            ("USA", "Iran", 0.05),
            
            # China relationships
            ("China", "Russia", 0.8),
            ("China", "Taiwan", 0.2),
            ("China", "Japan", 0.4),
            ("China", "South Korea", 0.6),
            ("China", "EU", 0.5),
            
            # Allied relationships
            ("Japan", "Taiwan", 0.8),
            ("Japan", "South Korea", 0.7),
            ("Taiwan", "South Korea", 0.75),
            ("Netherlands", "EU", 0.95),
            
            # Adversarial relationships
            ("Russia", "EU", 0.15),
            ("Russia", "Japan", 0.2),
            ("Iran", "EU", 0.25),
            ("Iran", "Japan", 0.3)
        ]
        
        for country1, country2, score in relationships:
            self.country_relationships[(country1, country2)] = score
            self.country_relationships[(country2, country1)] = score  # Symmetric
    
    def assess_export_compliance(self, technology_id: str, source_country: str, 
                                destination_country: str, end_user: str, 
                                quantity: float = 1.0, value_usd: float = 1000000) -> ComplianceAssessment:
        """Assess export compliance requirements for a technology transfer."""
        
        transaction_id = f"txn_{len(self.compliance_history)+1:06d}"
        
        # Get technology control data
        technology = self.controlled_technologies.get(technology_id)
        if not technology:
            return ComplianceAssessment(
                transaction_id=transaction_id,
                technology_id=technology_id,
                source_country=source_country,
                destination_country=destination_country,
                end_user_entity=end_user,
                compliance_status="unknown_technology",
                required_licenses=[],
                entity_list_flags=[],
                estimated_delay_days=0,
                compliance_cost_usd=0,
                risk_factors=["Unknown technology classification"]
            )
        
        # Check country controls
        license_required = False
        required_licenses = []
        
        if destination_country in technology.countries_controlled:
            license_req = technology.license_requirements.get(destination_country, 
                                                            technology.license_requirements.get("default"))
            if license_req != LicenseType.NO_LICENSE_REQUIRED:
                license_required = True
                required_licenses.append(license_req)
        
        # Check entity list status
        entity_flags = []
        source_entity_lists = self.entity_lists.get(source_country, {})
        entity_status = source_entity_lists.get(end_user, EntityListStatus.CLEARED)
        
        if entity_status != EntityListStatus.CLEARED:
            entity_flags.append(entity_status)
            if entity_status in [EntityListStatus.DENIED_PERSONS, EntityListStatus.ENTITY_LIST]:
                license_required = True
                if LicenseType.SPECIFIC not in required_licenses:
                    required_licenses.append(LicenseType.SPECIFIC)
        
        # Determine compliance status
        if entity_status == EntityListStatus.DENIED_PERSONS:
            compliance_status = "prohibited"
        elif (destination_country in technology.countries_controlled and 
              technology.license_requirements.get(destination_country) == LicenseType.PROHIBITED):
            compliance_status = "prohibited"
        elif license_required:
            compliance_status = "requires_license"
        else:
            compliance_status = "compliant"
        
        # Calculate delays and costs
        estimated_delay = self._calculate_license_processing_time(required_licenses, destination_country)
        compliance_cost = self._calculate_compliance_cost(technology, value_usd, required_licenses)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(technology, source_country, destination_country, 
                                                 end_user, entity_status)
        
        assessment = ComplianceAssessment(
            transaction_id=transaction_id,
            technology_id=technology_id,
            source_country=source_country,
            destination_country=destination_country,
            end_user_entity=end_user,
            compliance_status=compliance_status,
            required_licenses=required_licenses,
            entity_list_flags=entity_flags,
            estimated_delay_days=estimated_delay,
            compliance_cost_usd=compliance_cost,
            risk_factors=risk_factors
        )
        
        self.compliance_history.append(assessment)
        return assessment
    
    def _calculate_license_processing_time(self, required_licenses: List[LicenseType], 
                                         destination_country: str) -> int:
        """Calculate expected license processing time."""
        if not required_licenses:
            return 0
        
        base_times = {
            LicenseType.GENERAL: 0,
            LicenseType.SPECIFIC: 60,
            LicenseType.GLOBAL: 120,
            LicenseType.DEEMED_EXPORT: 90,
            LicenseType.LICENSE_EXCEPTION: 30
        }
        
        max_time = max(base_times.get(license, 60) for license in required_licenses)
        
        # Country-specific adjustments
        country_factors = {
            "China": 1.5,
            "Russia": 2.0,
            "Iran": 3.0,
            "North Korea": 3.0
        }
        
        factor = country_factors.get(destination_country, 1.0)
        return int(max_time * factor)
    
    def _calculate_compliance_cost(self, technology: ControlledTechnology, 
                                 value_usd: float, required_licenses: List[LicenseType]) -> float:
        """Calculate compliance cost including legal, administrative, and delay costs."""
        
        if not required_licenses:
            return 0
        
        # Base compliance costs
        license_costs = {
            LicenseType.GENERAL: 0,
            LicenseType.SPECIFIC: 25000,  # Legal and administrative
            LicenseType.GLOBAL: 50000,
            LicenseType.DEEMED_EXPORT: 35000,
            LicenseType.LICENSE_EXCEPTION: 15000
        }
        
        total_license_cost = sum(license_costs.get(license, 25000) for license in required_licenses)
        
        # Technology-specific cost multiplier
        tech_cost = value_usd * (technology.compliance_cost_factor - 1.0)
        
        # Opportunity cost from delays (assume 8% annual cost of capital)
        delay_days = self._calculate_license_processing_time(required_licenses, "default")
        opportunity_cost = value_usd * 0.08 * (delay_days / 365)
        
        return total_license_cost + tech_cost + opportunity_cost
    
    def _identify_risk_factors(self, technology: ControlledTechnology, source_country: str,
                             destination_country: str, end_user: str, 
                             entity_status: EntityListStatus) -> List[str]:
        """Identify compliance and business risk factors."""
        risks = []
        
        # Entity list risks
        if entity_status != EntityListStatus.CLEARED:
            risks.append(f"End user on {entity_status.value}")
        
        # Technology sensitivity risks
        if technology.control_level in [ControlLevel.RESTRICTED, ControlLevel.PROHIBITED]:
            risks.append("High sensitivity technology")
        
        # Country relationship risks
        relationship_score = self.country_relationships.get((source_country, destination_country), 0.5)
        if relationship_score < 0.3:
            risks.append("Poor bilateral relations")
        elif relationship_score < 0.6:
            risks.append("Strained bilateral relations")
        
        # End use risks
        high_risk_end_uses = ["military", "nuclear", "missile", "ai_training", "supercomputing"]
        for end_use in technology.end_use_restrictions:
            if end_use in high_risk_end_uses:
                risks.append(f"Restricted end use: {end_use}")
        
        # Geopolitical escalation risk
        if destination_country in ["China", "Russia", "Iran", "North Korea"]:
            risks.append("Geopolitical escalation risk")
        
        return risks
    
    def simulate_policy_change(self, policy_change: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of export control policy changes."""
        
        change_type = policy_change.get("type")
        effective_date = policy_change.get("effective_date", datetime.now())
        
        impact_analysis = {
            "policy_change": policy_change,
            "affected_technologies": [],
            "affected_countries": [],
            "trade_flow_impact": {},
            "compliance_cost_change": {},
            "strategic_implications": []
        }
        
        if change_type == "add_technology_control":
            # New technology added to control list
            tech_id = policy_change["technology_id"]
            new_controls = policy_change["controls"]
            
            if tech_id in self.controlled_technologies:
                # Update existing controls
                tech = self.controlled_technologies[tech_id]
                tech.countries_controlled.extend(new_controls.get("countries", []))
                tech.license_requirements.update(new_controls.get("license_requirements", {}))
            else:
                # Add new controlled technology
                # Implementation would create new ControlledTechnology object
                pass
            
            impact_analysis["affected_technologies"].append(tech_id)
            impact_analysis["strategic_implications"].append(
                f"New controls on {tech_id} may disrupt existing supply chains"
            )
        
        elif change_type == "entity_list_addition":
            # Add entity to control list
            controlling_country = policy_change["controlling_country"]
            entity_name = policy_change["entity_name"]
            list_status = EntityListStatus(policy_change["list_status"])
            
            if controlling_country not in self.entity_lists:
                self.entity_lists[controlling_country] = {}
            
            self.entity_lists[controlling_country][entity_name] = list_status
            
            impact_analysis["affected_countries"].append(controlling_country)
            impact_analysis["strategic_implications"].append(
                f"Addition of {entity_name} to {list_status.value} may impact technology access"
            )
        
        elif change_type == "sanctions_package":
            # Comprehensive sanctions on a country
            target_country = policy_change["target_country"]
            sanctions_scope = policy_change.get("scope", "comprehensive")
            
            # Update all relevant technology controls
            for tech in self.controlled_technologies.values():
                if target_country not in tech.countries_controlled:
                    tech.countries_controlled.append(target_country)
                    tech.license_requirements[target_country] = LicenseType.PROHIBITED
            
            impact_analysis["affected_countries"].append(target_country)
            impact_analysis["strategic_implications"].extend([
                f"Comprehensive sanctions on {target_country}",
                "Significant supply chain disruption expected",
                "Alternative sourcing strategies required"
            ])
        
        # Calculate economic impact
        impact_analysis["trade_flow_impact"] = self._estimate_trade_flow_impact(policy_change)
        impact_analysis["compliance_cost_change"] = self._estimate_compliance_cost_change(policy_change)
        
        # Store policy change
        self.policy_changes.append({
            "timestamp": effective_date,
            "policy": policy_change,
            "impact": impact_analysis
        })
        
        return impact_analysis
    
    def _estimate_trade_flow_impact(self, policy_change: Dict[str, Any]) -> Dict[str, float]:
        """Estimate impact on bilateral trade flows."""
        trade_impact = {}
        
        change_type = policy_change.get("type")
        
        if change_type == "sanctions_package":
            target_country = policy_change["target_country"]
            # Estimate trade reduction (simplified model)
            trade_impact[f"to_{target_country}"] = -0.8  # 80% reduction
            trade_impact[f"from_{target_country}"] = -0.7  # 70% reduction
            
        elif change_type == "add_technology_control":
            tech_id = policy_change["technology_id"]
            controlled_countries = policy_change.get("controls", {}).get("countries", [])
            
            for country in controlled_countries:
                trade_impact[f"{tech_id}_to_{country}"] = -0.3  # 30% reduction
        
        elif change_type == "entity_list_addition":
            entity_name = policy_change["entity_name"]
            trade_impact[f"entity_{entity_name}"] = -0.6  # 60% reduction in entity access
        
        return trade_impact
    
    def _estimate_compliance_cost_change(self, policy_change: Dict[str, Any]) -> Dict[str, float]:
        """Estimate change in compliance costs."""
        cost_changes = {}
        
        change_type = policy_change.get("type")
        
        if change_type == "add_technology_control":
            tech_id = policy_change["technology_id"]
            cost_changes[tech_id] = 50000  # Additional $50k compliance cost
            
        elif change_type == "entity_list_addition":
            cost_changes["entity_screening"] = 25000  # Additional screening costs
            
        elif change_type == "sanctions_package":
            cost_changes["sanctions_compliance"] = 100000  # Major compliance overhaul
        
        return cost_changes
    
    def analyze_technology_diffusion_impact(self, technology_id: str, 
                                          time_horizon_years: int = 5) -> Dict[str, Any]:
        """Analyze how export controls affect technology diffusion globally."""
        
        technology = self.controlled_technologies.get(technology_id)
        if not technology:
            return {"error": "Technology not found"}
        
        diffusion_analysis = {
            "technology_id": technology_id,
            "control_regime": technology.control_regime.value,
            "controlled_countries": technology.countries_controlled,
            "diffusion_delays": {},
            "alternative_development": {},
            "market_fragmentation": {},
            "innovation_impacts": []
        }
        
        # Calculate diffusion delays by country
        for country in technology.countries_controlled:
            license_type = technology.license_requirements.get(country)
            
            if license_type == LicenseType.PROHIBITED:
                delay_years = time_horizon_years  # Complete blocking
            elif license_type == LicenseType.SPECIFIC:
                delay_years = min(2.0, time_horizon_years * 0.4)  # Significant delay
            elif license_type == LicenseType.GENERAL:
                delay_years = min(0.5, time_horizon_years * 0.1)  # Minor delay
            else:
                delay_years = 0
            
            diffusion_analysis["diffusion_delays"][country] = delay_years
        
        # Estimate alternative development incentives
        for country in technology.countries_controlled:
            if technology.license_requirements.get(country) == LicenseType.PROHIBITED:
                # Strong incentive for indigenous development
                diffusion_analysis["alternative_development"][country] = {
                    "development_probability": 0.8,
                    "estimated_timeline_years": max(3, time_horizon_years * 0.6),
                    "investment_required_billions": self._estimate_development_cost(technology),
                    "technical_feasibility": self._assess_technical_feasibility(technology, country)
                }
        
        # Market fragmentation analysis
        controlled_market_share = 0.4  # Assume controlled countries represent 40% of market
        diffusion_analysis["market_fragmentation"] = {
            "fragmentation_risk": controlled_market_share,
            "alternative_ecosystems": len([c for c in technology.countries_controlled 
                                         if technology.license_requirements.get(c) == LicenseType.PROHIBITED]),
            "standards_divergence_risk": 0.3 if controlled_market_share > 0.3 else 0.1
        }
        
        # Innovation impact assessment
        if len(technology.countries_controlled) > 2:
            diffusion_analysis["innovation_impacts"].extend([
                "Reduced global collaboration on technology development",
                "Duplicated R&D efforts across geopolitical blocks",
                "Slower overall pace of innovation due to market fragmentation"
            ])
        
        if "China" in technology.countries_controlled:
            diffusion_analysis["innovation_impacts"].append(
                "Accelerated Chinese indigenous innovation programs"
            )
        
        return diffusion_analysis
    
    def _estimate_development_cost(self, technology: ControlledTechnology) -> float:
        """Estimate cost of indigenous technology development."""
        base_costs = {
            "euv_lithography": 50.0,  # $50B for EUV capability
            "advanced_semiconductors": 100.0,  # $100B for advanced fab capability
            "chip_design_software": 10.0,  # $10B for EDA tools
            "epitaxial_deposition": 5.0,   # $5B for equipment
            "ion_implantation": 3.0,       # $3B for equipment
            "rare_earth_elements": 1.0     # $1B for processing capability
        }
        
        return base_costs.get(technology.technology_id, 5.0)
    
    def _assess_technical_feasibility(self, technology: ControlledTechnology, country: str) -> float:
        """Assess technical feasibility of indigenous development."""
        
        # Country technical capabilities (0-1 scale)
        country_capabilities = {
            "China": 0.8,
            "Russia": 0.6,
            "Japan": 0.9,
            "South Korea": 0.85,
            "Taiwan": 0.9,
            "EU": 0.85,
            "USA": 0.95,
            "India": 0.7,
            "default": 0.5
        }
        
        base_capability = country_capabilities.get(country, country_capabilities["default"])
        
        # Technology complexity factors
        complexity_factors = {
            "euv_lithography": 0.3,  # Extremely difficult
            "advanced_semiconductors": 0.4,  # Very difficult
            "chip_design_software": 0.7,  # Moderately difficult
            "epitaxial_deposition": 0.8,   # Less difficult
            "ion_implantation": 0.8,       # Less difficult
            "rare_earth_elements": 0.9     # Relatively easier
        }
        
        complexity_factor = complexity_factors.get(technology.technology_id, 0.6)
        
        return base_capability * complexity_factor
    
    def get_export_control_summary(self) -> Dict[str, Any]:
        """Get comprehensive export control landscape summary."""
        
        # Technology control statistics
        total_technologies = len(self.controlled_technologies)
        by_regime = {}
        by_control_level = {}
        
        for tech in self.controlled_technologies.values():
            regime = tech.control_regime.value
            level = tech.control_level.value
            
            by_regime[regime] = by_regime.get(regime, 0) + 1
            by_control_level[level] = by_control_level.get(level, 0) + 1
        
        # Entity list statistics
        total_entities = sum(len(entities) for entities in self.entity_lists.values())
        entities_by_status = {}
        
        for country_entities in self.entity_lists.values():
            for status in country_entities.values():
                status_str = status.value
                entities_by_status[status_str] = entities_by_status.get(status_str, 0) + 1
        
        # Compliance assessment statistics
        total_assessments = len(self.compliance_history)
        compliance_outcomes = {}
        
        for assessment in self.compliance_history:
            outcome = assessment.compliance_status
            compliance_outcomes[outcome] = compliance_outcomes.get(outcome, 0) + 1
        
        # Calculate average compliance metrics
        if self.compliance_history:
            avg_delay = np.mean([a.estimated_delay_days for a in self.compliance_history])
            avg_cost = np.mean([a.compliance_cost_usd for a in self.compliance_history])
            total_compliance_cost = sum(a.compliance_cost_usd for a in self.compliance_history)
        else:
            avg_delay = 0
            avg_cost = 0
            total_compliance_cost = 0
        
        # Policy change impact
        recent_policy_changes = len([p for p in self.policy_changes 
                                   if p["timestamp"] > datetime.now() - timedelta(days=365)])
        
        return {
            "control_landscape": {
                "total_controlled_technologies": total_technologies,
                "technologies_by_regime": by_regime,
                "technologies_by_control_level": by_control_level,
                "most_restricted_countries": self._get_most_restricted_countries()
            },
            "entity_lists": {
                "total_listed_entities": total_entities,
                "entities_by_status": entities_by_status,
                "controlling_jurisdictions": list(self.entity_lists.keys())
            },
            "compliance_metrics": {
                "total_assessments": total_assessments,
                "compliance_outcomes": compliance_outcomes,
                "average_processing_delay_days": avg_delay,
                "average_compliance_cost_usd": avg_cost,
                "total_compliance_burden_usd": total_compliance_cost
            },
            "policy_dynamics": {
                "total_policy_changes": len(self.policy_changes),
                "recent_policy_changes": recent_policy_changes,
                "active_sanctions_regimes": self._count_active_sanctions()
            },
            "geopolitical_risk_assessment": self._assess_overall_geopolitical_risk()
        }
    
    def _get_most_restricted_countries(self) -> List[Tuple[str, int]]:
        """Get countries with most technology restrictions."""
        country_restriction_counts = {}
        
        for tech in self.controlled_technologies.values():
            for country in tech.countries_controlled:
                country_restriction_counts[country] = country_restriction_counts.get(country, 0) + 1
        
        # Sort by restriction count
        sorted_countries = sorted(country_restriction_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_countries[:10]
    
    def _count_active_sanctions(self) -> int:
        """Count active sanctions regimes."""
        sanctioned_countries = set()
        
        for tech in self.controlled_technologies.values():
            for country, license_type in tech.license_requirements.items():
                if license_type == LicenseType.PROHIBITED:
                    sanctioned_countries.add(country)
        
        return len(sanctioned_countries)
    
    def _assess_overall_geopolitical_risk(self) -> Dict[str, float]:
        """Assess overall geopolitical risk in export control landscape."""
        
        # Calculate relationship deterioration trend
        poor_relationships = sum(1 for score in self.country_relationships.values() if score < 0.3)
        total_relationships = len(self.country_relationships)
        
        relationship_risk = poor_relationships / max(total_relationships, 1)
        
        # Calculate control escalation risk
        recent_escalations = len([p for p in self.policy_changes 
                                if p["timestamp"] > datetime.now() - timedelta(days=180)])
        escalation_risk = min(recent_escalations / 10, 1.0)  # Normalize
        
        # Calculate technology competition risk
        critical_tech_controls = sum(1 for tech in self.controlled_technologies.values() 
                                   if tech.control_level in [ControlLevel.RESTRICTED, ControlLevel.PROHIBITED])
        
        tech_competition_risk = critical_tech_controls / max(len(self.controlled_technologies), 1)
        
        return {
            "relationship_deterioration_risk": relationship_risk,
            "control_escalation_risk": escalation_risk,
            "technology_competition_risk": tech_competition_risk,
            "overall_risk_score": (relationship_risk + escalation_risk + tech_competition_risk) / 3
        } 