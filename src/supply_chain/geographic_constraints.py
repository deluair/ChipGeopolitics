"""
Geographic Constraints Model for Semiconductor Supply Chains

Models geographic and logistical constraints affecting semiconductor supply chains
including trade routes, shipping infrastructure, geopolitical barriers, and regional dependencies.
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import geopy.distance
from datetime import datetime, timedelta

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class TransportMode(Enum):
    """Transportation modes for supply chain."""
    AIR_FREIGHT = "air_freight"
    OCEAN_FREIGHT = "ocean_freight"
    RAIL_FREIGHT = "rail_freight"
    TRUCK_FREIGHT = "truck_freight"
    PIPELINE = "pipeline"

class InfrastructureType(Enum):
    """Types of transportation infrastructure."""
    AIRPORT = "airport"
    SEAPORT = "seaport"
    RAIL_TERMINAL = "rail_terminal"
    HIGHWAY = "highway"
    WAREHOUSE = "warehouse"
    CUSTOMS_FACILITY = "customs_facility"

class GeopoliticalBarrier(Enum):
    """Types of geopolitical barriers."""
    TRADE_SANCTIONS = "trade_sanctions"
    EXPORT_CONTROLS = "export_controls"
    TARIFFS = "tariffs"
    QUOTA_RESTRICTIONS = "quota_restrictions"
    LICENSING_REQUIREMENTS = "licensing_requirements"
    BORDER_RESTRICTIONS = "border_restrictions"

@dataclass
class GeographicLocation:
    """Geographic location with coordinates and attributes."""
    location_id: str
    name: str
    country: str
    region: str
    latitude: float
    longitude: float
    time_zone: str
    infrastructure_quality: float  # 0-1 quality score
    political_stability: float  # 0-1 stability score
    trade_facilitation_index: float  # 0-1 ease of trade

@dataclass
class TransportRoute:
    """Transportation route between locations."""
    route_id: str
    origin: str
    destination: str
    transport_mode: TransportMode
    distance_km: float
    transit_time_hours: float
    cost_per_kg: float
    reliability: float  # 0-1 on-time performance
    capacity_tonnes_per_day: float
    infrastructure_dependencies: List[str]
    seasonal_constraints: Dict[str, float]  # Month -> capacity factor

@dataclass
class GeopoliticalConstraint:
    """Geopolitical constraint affecting trade."""
    constraint_id: str
    barrier_type: GeopoliticalBarrier
    origin_countries: List[str]
    destination_countries: List[str]
    affected_products: List[str]
    severity: float  # 0-1 impact on trade flow
    implementation_date: datetime
    expiration_date: Optional[datetime]
    compliance_cost: float  # Additional cost per unit

@dataclass
class InfrastructureNode:
    """Infrastructure facility."""
    facility_id: str
    facility_type: InfrastructureType
    location: str
    capacity: float
    utilization_rate: float
    upgrade_cost: float
    criticality_score: float  # 0-1 importance to network

class GeographicConstraintModel:
    """
    Comprehensive geographic constraints modeling for semiconductor supply chains.
    
    Models:
    - Transportation routes and modal choices
    - Infrastructure capacity and bottlenecks
    - Geopolitical trade barriers and sanctions
    - Shipping time optimization
    - Regional dependency analysis
    - Climate and seasonal impacts
    """
    
    def __init__(self):
        # Geographic data
        self.locations: Dict[str, GeographicLocation] = {}
        self.transport_routes: Dict[str, TransportRoute] = {}
        self.infrastructure_nodes: Dict[str, InfrastructureNode] = {}
        
        # Constraints and barriers
        self.geopolitical_constraints: Dict[str, GeopoliticalConstraint] = {}
        self.active_disruptions: Dict[str, Dict[str, Any]] = {}
        
        # Analysis results
        self.route_optimization_results: Dict[str, List[str]] = {}
        self.dependency_analysis: Dict[str, float] = {}
        self.vulnerability_assessment: Dict[str, Dict[str, float]] = {}
        
        # Initialize with realistic geographic data
        self._initialize_geographic_data()
        self._initialize_transport_routes()
        self._initialize_infrastructure()
        self._initialize_geopolitical_constraints()
    
    def _initialize_geographic_data(self):
        """Initialize realistic geographic locations."""
        
        # Key semiconductor industry locations with coordinates
        locations_data = [
            # Asia-Pacific
            ("taiwan_north", "Northern Taiwan", "Taiwan", "Asia-Pacific", 25.0330, 121.5654, "UTC+8", 0.95, 0.85, 0.90),
            ("taiwan_south", "Southern Taiwan", "Taiwan", "Asia-Pacific", 22.6273, 120.3014, "UTC+8", 0.93, 0.85, 0.88),
            ("seoul", "Seoul", "South Korea", "Asia-Pacific", 37.5665, 126.9780, "UTC+9", 0.92, 0.80, 0.85),
            ("tokyo", "Tokyo", "Japan", "Asia-Pacific", 35.6762, 139.6503, "UTC+9", 0.96, 0.90, 0.92),
            ("shanghai", "Shanghai", "China", "Asia-Pacific", 31.2304, 121.4737, "UTC+8", 0.88, 0.75, 0.82),
            ("shenzhen", "Shenzhen", "China", "Asia-Pacific", 22.5431, 114.0579, "UTC+8", 0.85, 0.75, 0.80),
            ("singapore", "Singapore", "Singapore", "Asia-Pacific", 1.3521, 103.8198, "UTC+8", 0.98, 0.95, 0.98),
            
            # North America
            ("silicon_valley", "Silicon Valley", "USA", "North America", 37.3861, -122.0839, "UTC-8", 0.95, 0.90, 0.95),
            ("austin", "Austin", "USA", "North America", 30.2672, -97.7431, "UTC-6", 0.90, 0.88, 0.90),
            ("portland", "Portland", "USA", "North America", 45.5152, -122.6784, "UTC-8", 0.88, 0.85, 0.88),
            ("phoenix", "Phoenix", "USA", "North America", 33.4484, -112.0740, "UTC-7", 0.85, 0.82, 0.85),
            
            # Europe
            ("amsterdam", "Amsterdam", "Netherlands", "Europe", 52.3676, 4.9041, "UTC+1", 0.94, 0.92, 0.95),
            ("dresden", "Dresden", "Germany", "Europe", 51.0504, 13.7373, "UTC+1", 0.92, 0.88, 0.90),
            ("grenoble", "Grenoble", "France", "Europe", 45.1885, 5.7245, "UTC+1", 0.90, 0.85, 0.88),
            ("cork", "Cork", "Ireland", "Europe", 51.8985, -8.4756, "UTC+0", 0.88, 0.90, 0.92)
        ]
        
        for location_data in locations_data:
            location = GeographicLocation(*location_data)
            self.locations[location.location_id] = location
    
    def _initialize_transport_routes(self):
        """Initialize realistic transportation routes."""
        
        # Calculate routes between major semiconductor locations
        route_data = []
        
        # Key origin-destination pairs with realistic transport options
        od_pairs = [
            # Taiwan to major markets
            ("taiwan_north", "silicon_valley", TransportMode.AIR_FREIGHT, 11000, 14, 8.5, 0.92, 50),
            ("taiwan_north", "silicon_valley", TransportMode.OCEAN_FREIGHT, 11500, 336, 2.1, 0.85, 500),
            ("taiwan_south", "austin", TransportMode.AIR_FREIGHT, 13500, 18, 9.2, 0.90, 45),
            
            # Korea to global markets  
            ("seoul", "silicon_valley", TransportMode.AIR_FREIGHT, 8800, 12, 7.8, 0.94, 60),
            ("seoul", "amsterdam", TransportMode.AIR_FREIGHT, 8200, 11, 8.0, 0.91, 40),
            ("seoul", "amsterdam", TransportMode.OCEAN_FREIGHT, 19000, 720, 1.8, 0.88, 800),
            
            # Japan routes
            ("tokyo", "silicon_valley", TransportMode.AIR_FREIGHT, 8300, 11, 7.5, 0.96, 70),
            ("tokyo", "amsterdam", TransportMode.AIR_FREIGHT, 9600, 13, 8.2, 0.93, 55),
            ("tokyo", "portland", TransportMode.OCEAN_FREIGHT, 7800, 240, 1.5, 0.90, 1000),
            
            # China routes
            ("shanghai", "silicon_valley", TransportMode.AIR_FREIGHT, 11000, 14, 6.8, 0.88, 80),
            ("shanghai", "amsterdam", TransportMode.RAIL_FREIGHT, 11000, 360, 3.2, 0.82, 200),
            ("shenzhen", "austin", TransportMode.OCEAN_FREIGHT, 16500, 480, 2.5, 0.85, 600),
            
            # Intra-Asia routes
            ("taiwan_north", "singapore", TransportMode.AIR_FREIGHT, 2300, 4, 4.5, 0.95, 30),
            ("seoul", "shanghai", TransportMode.OCEAN_FREIGHT, 950, 48, 1.2, 0.90, 300),
            ("tokyo", "taiwan_north", TransportMode.AIR_FREIGHT, 2200, 3, 5.2, 0.94, 40),
            
            # Europe internal
            ("amsterdam", "dresden", TransportMode.TRUCK_FREIGHT, 650, 8, 2.8, 0.95, 25),
            ("amsterdam", "grenoble", TransportMode.TRUCK_FREIGHT, 1050, 12, 3.1, 0.92, 20),
            
            # US internal
            ("silicon_valley", "austin", TransportMode.AIR_FREIGHT, 2400, 4, 6.0, 0.93, 35),
            ("portland", "phoenix", TransportMode.TRUCK_FREIGHT, 1800, 20, 2.2, 0.88, 40)
        ]
        
        # Generate route objects
        for i, (origin, dest, mode, distance, time, cost, reliability, capacity) in enumerate(od_pairs):
            route_id = f"route_{i+1:03d}_{origin}_{dest}_{mode.value}"
            
            # Add seasonal constraints
            seasonal_constraints = {}
            if mode == TransportMode.OCEAN_FREIGHT:
                # Ocean routes affected by weather
                seasonal_constraints = {
                    "1": 0.85, "2": 0.85, "3": 0.90, "4": 0.95, "5": 1.0, "6": 1.0,
                    "7": 1.0, "8": 1.0, "9": 0.95, "10": 0.90, "11": 0.85, "12": 0.80
                }
            elif mode == TransportMode.RAIL_FREIGHT:
                # Rail slightly affected by winter weather
                seasonal_constraints = {
                    "1": 0.90, "2": 0.90, "3": 0.95, "4": 1.0, "5": 1.0, "6": 1.0,
                    "7": 1.0, "8": 1.0, "9": 1.0, "10": 0.95, "11": 0.90, "12": 0.90
                }
            else:
                # Air and truck less affected
                seasonal_constraints = {str(m): 0.95 for m in range(1, 13)}
            
            # Infrastructure dependencies
            infrastructure_deps = []
            if mode == TransportMode.AIR_FREIGHT:
                infrastructure_deps = [f"{origin}_airport", f"{dest}_airport"]
            elif mode == TransportMode.OCEAN_FREIGHT:
                infrastructure_deps = [f"{origin}_port", f"{dest}_port"]
            elif mode == TransportMode.RAIL_FREIGHT:
                infrastructure_deps = [f"{origin}_rail", f"{dest}_rail"]
            elif mode == TransportMode.TRUCK_FREIGHT:
                infrastructure_deps = [f"{origin}_highway", f"{dest}_highway"]
            
            route = TransportRoute(
                route_id=route_id,
                origin=origin,
                destination=dest,
                transport_mode=mode,
                distance_km=distance,
                transit_time_hours=time,
                cost_per_kg=cost,
                reliability=reliability,
                capacity_tonnes_per_day=capacity,
                infrastructure_dependencies=infrastructure_deps,
                seasonal_constraints=seasonal_constraints
            )
            
            self.transport_routes[route_id] = route
    
    def _initialize_infrastructure(self):
        """Initialize infrastructure facilities."""
        
        infrastructure_data = [
            # Airports
            ("taiwan_north_airport", InfrastructureType.AIRPORT, "taiwan_north", 1000, 0.85, 50_000_000, 0.95),
            ("seoul_airport", InfrastructureType.AIRPORT, "seoul", 800, 0.80, 45_000_000, 0.90),
            ("tokyo_airport", InfrastructureType.AIRPORT, "tokyo", 1200, 0.88, 60_000_000, 0.92),
            ("shanghai_airport", InfrastructureType.AIRPORT, "shanghai", 900, 0.75, 40_000_000, 0.85),
            ("silicon_valley_airport", InfrastructureType.AIRPORT, "silicon_valley", 600, 0.82, 35_000_000, 0.88),
            ("amsterdam_airport", InfrastructureType.AIRPORT, "amsterdam", 1100, 0.90, 55_000_000, 0.96),
            
            # Seaports
            ("taiwan_north_port", InfrastructureType.SEAPORT, "taiwan_north", 5000, 0.92, 100_000_000, 0.98),
            ("seoul_port", InfrastructureType.SEAPORT, "seoul", 4000, 0.85, 80_000_000, 0.90),
            ("tokyo_port", InfrastructureType.SEAPORT, "tokyo", 3500, 0.80, 90_000_000, 0.88),
            ("shanghai_port", InfrastructureType.SEAPORT, "shanghai", 8000, 0.95, 120_000_000, 0.95),
            ("singapore_port", InfrastructureType.SEAPORT, "singapore", 6000, 0.98, 110_000_000, 0.99),
            
            # Warehouses and distribution centers
            ("taiwan_north_warehouse", InfrastructureType.WAREHOUSE, "taiwan_north", 2000, 0.88, 20_000_000, 0.85),
            ("silicon_valley_warehouse", InfrastructureType.WAREHOUSE, "silicon_valley", 1500, 0.90, 25_000_000, 0.82),
            ("amsterdam_warehouse", InfrastructureType.WAREHOUSE, "amsterdam", 1800, 0.85, 22_000_000, 0.80),
            
            # Rail terminals
            ("shanghai_rail", InfrastructureType.RAIL_TERMINAL, "shanghai", 800, 0.80, 15_000_000, 0.75),
            ("amsterdam_rail", InfrastructureType.RAIL_TERMINAL, "amsterdam", 600, 0.88, 18_000_000, 0.78),
            ("dresden_rail", InfrastructureType.RAIL_TERMINAL, "dresden", 400, 0.85, 12_000_000, 0.70)
        ]
        
        for facility_data in infrastructure_data:
            facility = InfrastructureNode(*facility_data)
            self.infrastructure_nodes[facility.facility_id] = facility
    
    def _initialize_geopolitical_constraints(self):
        """Initialize current geopolitical constraints."""
        
        # Current trade tensions and export controls (as of 2024)
        constraints_data = [
            # US-China technology controls
            ("us_china_chips", GeopoliticalBarrier.EXPORT_CONTROLS, ["USA"], ["China"], 
             ["advanced_semiconductors", "chip_equipment"], 0.8, datetime(2022, 10, 1), None, 15.0),
            
            # China rare earth restrictions
            ("china_rare_earth", GeopoliticalBarrier.EXPORT_CONTROLS, ["China"], ["USA"], 
             ["rare_earth_materials"], 0.6, datetime(2023, 8, 1), None, 8.0),
            
            # Russia sanctions
            ("russia_sanctions", GeopoliticalBarrier.TRADE_SANCTIONS, ["Russia"], ["USA", "EU", "Japan"], 
             ["all_semiconductors"], 0.95, datetime(2022, 2, 24), None, 25.0),
            
            # Taiwan Strait tensions - increased shipping insurance
            ("taiwan_strait_risk", GeopoliticalBarrier.BORDER_RESTRICTIONS, ["Taiwan"], ["all"], 
             ["all_semiconductors"], 0.3, datetime(2024, 1, 1), None, 5.0),
            
            # EU-China investment screening
            ("eu_china_screening", GeopoliticalBarrier.LICENSING_REQUIREMENTS, ["China"], ["EU"], 
             ["semiconductor_equipment"], 0.4, datetime(2023, 6, 1), None, 3.0)
        ]
        
        for constraint_data in constraints_data:
            constraint_id, barrier_type, origins, destinations, products, severity, start_date, end_date, cost = constraint_data
            
            constraint = GeopoliticalConstraint(
                constraint_id=constraint_id,
                barrier_type=barrier_type,
                origin_countries=origins,
                destination_countries=destinations,
                affected_products=products,
                severity=severity,
                implementation_date=start_date,
                expiration_date=end_date,
                compliance_cost=cost
            )
            
            self.geopolitical_constraints[constraint_id] = constraint
    
    def optimize_shipping_routes(self, origin: str, destination: str, 
                                cargo_type: str = "semiconductors", 
                                weight_kg: float = 1000,
                                priority: str = "cost") -> Dict[str, Any]:
        """Optimize shipping routes between locations."""
        
        # Find all possible routes
        possible_routes = []
        for route in self.transport_routes.values():
            if route.origin == origin and route.destination == destination:
                possible_routes.append(route)
        
        if not possible_routes:
            # Try to find indirect routes
            possible_routes = self._find_indirect_routes(origin, destination)
        
        if not possible_routes:
            return {
                'status': 'no_route_found',
                'origin': origin,
                'destination': destination,
                'message': 'No viable routes found'
            }
        
        # Calculate route scores based on priority
        route_evaluations = []
        for route in possible_routes:
            evaluation = self._evaluate_route(route, cargo_type, weight_kg, priority)
            route_evaluations.append({
                'route': route,
                'evaluation': evaluation
            })
        
        # Sort by optimization criterion
        if priority == "cost":
            route_evaluations.sort(key=lambda x: x['evaluation']['total_cost'])
        elif priority == "time":
            route_evaluations.sort(key=lambda x: x['evaluation']['total_time'])
        elif priority == "reliability":
            route_evaluations.sort(key=lambda x: x['evaluation']['reliability_score'], reverse=True)
        else:  # Balanced
            route_evaluations.sort(key=lambda x: x['evaluation']['composite_score'], reverse=True)
        
        # Prepare results
        best_route = route_evaluations[0]
        alternatives = route_evaluations[1:4]  # Top 3 alternatives
        
        result = {
            'status': 'success',
            'origin': origin,
            'destination': destination,
            'recommended_route': {
                'route_id': best_route['route']['route_id'],
                'transport_mode': best_route['route']['transport_mode'].value,
                'total_cost_usd': best_route['evaluation']['total_cost'],
                'transit_time_hours': best_route['evaluation']['total_time'],
                'reliability_score': best_route['evaluation']['reliability_score'],
                'geopolitical_risk': best_route['evaluation']['geopolitical_risk'],
                'infrastructure_dependencies': best_route['route']['infrastructure_dependencies']
            },
            'alternative_routes': [
                {
                    'route_id': alt['route']['route_id'],
                    'transport_mode': alt['route']['transport_mode'].value,
                    'total_cost_usd': alt['evaluation']['total_cost'],
                    'transit_time_hours': alt['evaluation']['total_time'],
                    'reliability_score': alt['evaluation']['reliability_score']
                } for alt in alternatives
            ],
            'risk_factors': self._analyze_route_risks(best_route['route'], cargo_type),
            'optimization_notes': self._generate_route_recommendations(best_route['route'], cargo_type)
        }
        
        return result
    
    def _find_indirect_routes(self, origin: str, destination: str) -> List[TransportRoute]:
        """Find indirect routes through intermediate hubs."""
        indirect_routes = []
        
        # Common transshipment hubs
        hubs = ["singapore", "amsterdam", "tokyo", "shanghai"]
        
        for hub in hubs:
            if hub != origin and hub != destination:
                # Check for origin -> hub route
                origin_hub_routes = [r for r in self.transport_routes.values() 
                                   if r.origin == origin and r.destination == hub]
                
                # Check for hub -> destination route
                hub_dest_routes = [r for r in self.transport_routes.values() 
                                 if r.origin == hub and r.destination == destination]
                
                # Create combined routes
                for oh_route in origin_hub_routes:
                    for hd_route in hub_dest_routes:
                        # Create composite route
                        combined_route = self._create_combined_route(oh_route, hd_route, hub)
                        indirect_routes.append(combined_route)
        
        return indirect_routes
    
    def _create_combined_route(self, route1: TransportRoute, route2: TransportRoute, hub: str) -> TransportRoute:
        """Create a combined route through a hub."""
        route_id = f"combined_{route1.route_id}_{route2.route_id}"
        
        # Combined metrics
        total_distance = route1.distance_km + route2.distance_km
        total_time = route1.transit_time_hours + route2.transit_time_hours + 24  # 24h transshipment time
        combined_cost = route1.cost_per_kg + route2.cost_per_kg + 2.0  # Handling cost
        combined_reliability = route1.reliability * route2.reliability * 0.95  # Transshipment risk
        min_capacity = min(route1.capacity_tonnes_per_day, route2.capacity_tonnes_per_day)
        
        # Combined infrastructure dependencies
        combined_deps = route1.infrastructure_dependencies + route2.infrastructure_dependencies
        
        return TransportRoute(
            route_id=route_id,
            origin=route1.origin,
            destination=route2.destination,
            transport_mode=route1.transport_mode,  # Use first leg mode as primary
            distance_km=total_distance,
            transit_time_hours=total_time,
            cost_per_kg=combined_cost,
            reliability=combined_reliability,
            capacity_tonnes_per_day=min_capacity,
            infrastructure_dependencies=list(set(combined_deps)),
            seasonal_constraints=route1.seasonal_constraints  # Use first leg constraints
        )
    
    def _evaluate_route(self, route: TransportRoute, cargo_type: str, weight_kg: float, priority: str) -> Dict[str, float]:
        """Evaluate a route based on multiple criteria."""
        
        # Base costs and times
        base_cost = route.cost_per_kg * weight_kg
        base_time = route.transit_time_hours
        base_reliability = route.reliability
        
        # Apply geopolitical adjustments
        geo_risk, geo_cost_multiplier = self._calculate_geopolitical_impact(route, cargo_type)
        
        # Apply seasonal adjustments
        current_month = str(datetime.now().month)
        seasonal_factor = route.seasonal_constraints.get(current_month, 1.0)
        
        # Apply infrastructure utilization impacts
        infrastructure_delay = self._calculate_infrastructure_delays(route)
        
        # Final calculations
        total_cost = base_cost * geo_cost_multiplier * (2 - seasonal_factor)  # Higher cost in low season
        total_time = base_time + infrastructure_delay
        reliability_score = base_reliability * seasonal_factor * (1 - geo_risk)
        
        # Composite score for balanced optimization
        if priority == "balanced":
            # Normalize metrics to 0-1 scale
            cost_score = max(0, 1 - (total_cost / (weight_kg * 20)))  # Assume $20/kg as high cost
            time_score = max(0, 1 - (total_time / (7 * 24)))  # 7 days as long time
            
            composite_score = (cost_score * 0.4 + time_score * 0.3 + reliability_score * 0.3)
        else:
            composite_score = reliability_score  # Fallback
        
        return {
            'total_cost': total_cost,
            'total_time': total_time,
            'reliability_score': reliability_score,
            'geopolitical_risk': geo_risk,
            'seasonal_factor': seasonal_factor,
            'infrastructure_delay': infrastructure_delay,
            'composite_score': composite_score
        }
    
    def _calculate_geopolitical_impact(self, route: TransportRoute, cargo_type: str) -> Tuple[float, float]:
        """Calculate geopolitical risk and cost impact."""
        risk_score = 0.0
        cost_multiplier = 1.0
        
        origin_location = self.locations.get(route.origin)
        dest_location = self.locations.get(route.destination)
        
        if not origin_location or not dest_location:
            return 0.1, 1.1  # Default minimal risk
        
        origin_country = origin_location.country
        dest_country = dest_location.country
        
        # Check active constraints
        for constraint in self.geopolitical_constraints.values():
            if self._constraint_applies(constraint, origin_country, dest_country, cargo_type):
                risk_impact = constraint.severity * 0.5  # Convert to risk score
                cost_impact = 1 + (constraint.compliance_cost / 100)  # Convert percentage to multiplier
                
                risk_score = max(risk_score, risk_impact)
                cost_multiplier = max(cost_multiplier, cost_impact)
        
        return min(risk_score, 1.0), cost_multiplier
    
    def _constraint_applies(self, constraint: GeopoliticalConstraint, origin_country: str, 
                          dest_country: str, cargo_type: str) -> bool:
        """Check if a geopolitical constraint applies to this trade."""
        
        # Check countries
        origin_match = any(origin_country in origins for origins in [constraint.origin_countries])
        dest_match = any(dest_country in dests for dests in [constraint.destination_countries])
        
        if not (origin_match and dest_match):
            return False
        
        # Check products
        if "all" in constraint.affected_products[0] or cargo_type in constraint.affected_products:
            return True
        
        # Check if current date is within constraint period
        current_date = datetime.now()
        if constraint.implementation_date > current_date:
            return False
        
        if constraint.expiration_date and constraint.expiration_date < current_date:
            return False
        
        return True
    
    def _calculate_infrastructure_delays(self, route: TransportRoute) -> float:
        """Calculate delays due to infrastructure utilization."""
        total_delay = 0.0
        
        for infra_id in route.infrastructure_dependencies:
            infrastructure = self.infrastructure_nodes.get(infra_id)
            if infrastructure:
                # Delay increases with utilization rate
                if infrastructure.utilization_rate > 0.9:
                    delay_hours = (infrastructure.utilization_rate - 0.9) * 48  # Up to 48h delay
                elif infrastructure.utilization_rate > 0.8:
                    delay_hours = (infrastructure.utilization_rate - 0.8) * 12  # Up to 12h delay
                else:
                    delay_hours = 0
                
                total_delay += delay_hours
        
        return total_delay
    
    def _analyze_route_risks(self, route: TransportRoute, cargo_type: str) -> List[Dict[str, Any]]:
        """Analyze risks for a specific route."""
        risks = []
        
        # Infrastructure dependency risks
        for infra_id in route.infrastructure_dependencies:
            infrastructure = self.infrastructure_nodes.get(infra_id)
            if infrastructure and infrastructure.utilization_rate > 0.85:
                risks.append({
                    'type': 'infrastructure_congestion',
                    'description': f"High utilization at {infrastructure.facility_type.value}",
                    'probability': infrastructure.utilization_rate,
                    'impact': 'medium'
                })
        
        # Geopolitical risks
        geo_risk, _ = self._calculate_geopolitical_impact(route, cargo_type)
        if geo_risk > 0.3:
            risks.append({
                'type': 'geopolitical',
                'description': 'Trade restrictions or political tensions may affect shipment',
                'probability': geo_risk,
                'impact': 'high'
            })
        
        # Seasonal risks
        current_month = str(datetime.now().month)
        seasonal_factor = route.seasonal_constraints.get(current_month, 1.0)
        if seasonal_factor < 0.9:
            risks.append({
                'type': 'seasonal',
                'description': 'Weather conditions may cause delays',
                'probability': 1 - seasonal_factor,
                'impact': 'medium'
            })
        
        # Transport mode specific risks
        if route.transport_mode == TransportMode.OCEAN_FREIGHT:
            risks.append({
                'type': 'maritime',
                'description': 'Port congestion and weather delays possible',
                'probability': 0.2,
                'impact': 'medium'
            })
        elif route.transport_mode == TransportMode.AIR_FREIGHT:
            risks.append({
                'type': 'aviation',
                'description': 'Flight cancellations and capacity constraints',
                'probability': 0.1,
                'impact': 'low'
            })
        
        return risks
    
    def _generate_route_recommendations(self, route: TransportRoute, cargo_type: str) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Insurance recommendations
        geo_risk, _ = self._calculate_geopolitical_impact(route, cargo_type)
        if geo_risk > 0.4:
            recommendations.append("Consider purchasing political risk insurance")
        
        if route.transport_mode == TransportMode.OCEAN_FREIGHT:
            recommendations.append("Consider marine cargo insurance for ocean transit")
        
        # Timing recommendations
        current_month = str(datetime.now().month)
        seasonal_factor = route.seasonal_constraints.get(current_month, 1.0)
        if seasonal_factor < 0.9:
            recommendations.append("Consider shipping in more favorable season")
        
        # Alternative route recommendations
        if route.reliability < 0.85:
            recommendations.append("Consider backup shipping options due to reliability concerns")
        
        # Inventory recommendations
        if route.transit_time_hours > 72:  # More than 3 days
            recommendations.append("Consider safety stock to buffer long transit times")
        
        return recommendations
    
    def analyze_regional_dependencies(self) -> Dict[str, Dict[str, float]]:
        """Analyze regional dependencies in the supply chain."""
        dependencies = {}
        
        # Analyze by region
        regions = set(loc.region for loc in self.locations.values())
        
        for region in regions:
            region_analysis = {
                'production_capacity': 0.0,
                'consumption_demand': 0.0,
                'export_dependency': 0.0,
                'import_dependency': 0.0,
                'infrastructure_quality': 0.0,
                'geopolitical_risk': 0.0
            }
            
            region_locations = [loc for loc in self.locations.values() if loc.region == region]
            
            if region_locations:
                # Average infrastructure quality
                region_analysis['infrastructure_quality'] = np.mean([loc.infrastructure_quality for loc in region_locations])
                
                # Average political stability (inverse of geopolitical risk)
                region_analysis['geopolitical_risk'] = 1 - np.mean([loc.political_stability for loc in region_locations])
                
                # Export/import flow analysis
                export_routes = [route for route in self.transport_routes.values() 
                               if any(route.origin == loc.location_id for loc in region_locations)]
                import_routes = [route for route in self.transport_routes.values() 
                               if any(route.destination == loc.location_id for loc in region_locations)]
                
                total_export_capacity = sum(route.capacity_tonnes_per_day for route in export_routes)
                total_import_capacity = sum(route.capacity_tonnes_per_day for route in import_routes)
                
                region_analysis['export_dependency'] = min(total_export_capacity / 1000, 1.0)  # Normalize
                region_analysis['import_dependency'] = min(total_import_capacity / 1000, 1.0)
            
            dependencies[region] = region_analysis
        
        self.dependency_analysis = dependencies
        return dependencies
    
    def assess_infrastructure_vulnerabilities(self) -> Dict[str, Dict[str, float]]:
        """Assess vulnerabilities in transportation infrastructure."""
        vulnerabilities = {}
        
        for facility_id, facility in self.infrastructure_nodes.items():
            vulnerability_score = 0.0
            
            # Utilization risk
            utilization_risk = max(0, facility.utilization_rate - 0.8) / 0.2
            
            # Single point of failure risk
            dependent_routes = [route for route in self.transport_routes.values() 
                              if facility_id in route.infrastructure_dependencies]
            
            dependency_risk = min(len(dependent_routes) / 10, 1.0)
            
            # Location risk
            location = self.locations.get(facility.location)
            if location:
                location_risk = 1 - location.political_stability
            else:
                location_risk = 0.5
            
            # Combined vulnerability
            vulnerability_score = (utilization_risk * 0.4 + dependency_risk * 0.3 + location_risk * 0.3)
            
            vulnerabilities[facility_id] = {
                'overall_vulnerability': vulnerability_score,
                'utilization_risk': utilization_risk,
                'dependency_risk': dependency_risk,
                'location_risk': location_risk,
                'criticality_score': facility.criticality_score,
                'upgrade_cost_millions': facility.upgrade_cost / 1_000_000
            }
        
        self.vulnerability_assessment = vulnerabilities
        return vulnerabilities
    
    def get_geographic_summary(self) -> Dict[str, Any]:
        """Get comprehensive geographic constraints summary."""
        
        # Ensure analyses are complete
        if not self.dependency_analysis:
            self.analyze_regional_dependencies()
        
        if not self.vulnerability_assessment:
            self.assess_infrastructure_vulnerabilities()
        
        # Summary statistics
        total_routes = len(self.transport_routes)
        active_constraints = len([c for c in self.geopolitical_constraints.values() 
                                if c.expiration_date is None or c.expiration_date > datetime.now()])
        
        # Route mode distribution
        mode_distribution = {}
        for route in self.transport_routes.values():
            mode = route.transport_mode.value
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        
        # Infrastructure utilization
        high_utilization_facilities = sum(1 for facility in self.infrastructure_nodes.values() 
                                        if facility.utilization_rate > 0.85)
        
        # Geographic concentration
        country_concentration = {}
        for location in self.locations.values():
            country = location.country
            country_concentration[country] = country_concentration.get(country, 0) + 1
        
        return {
            'network_overview': {
                'total_locations': len(self.locations),
                'total_routes': total_routes,
                'infrastructure_facilities': len(self.infrastructure_nodes),
                'active_geopolitical_constraints': active_constraints
            },
            'route_analysis': {
                'transport_mode_distribution': mode_distribution,
                'average_route_reliability': np.mean([route.reliability for route in self.transport_routes.values()]),
                'total_daily_capacity_tonnes': sum(route.capacity_tonnes_per_day for route in self.transport_routes.values())
            },
            'regional_dependencies': self.dependency_analysis,
            'infrastructure_status': {
                'high_utilization_facilities': high_utilization_facilities,
                'average_utilization_rate': np.mean([facility.utilization_rate for facility in self.infrastructure_nodes.values()]),
                'total_upgrade_cost_billions': sum(facility.upgrade_cost for facility in self.infrastructure_nodes.values()) / 1_000_000_000
            },
            'geopolitical_landscape': {
                'active_constraints_by_type': {barrier.value: sum(1 for c in self.geopolitical_constraints.values() if c.barrier_type == barrier) 
                                            for barrier in GeopoliticalBarrier},
                'affected_trade_corridors': self._count_affected_trade_corridors(),
                'average_compliance_cost': np.mean([c.compliance_cost for c in self.geopolitical_constraints.values()])
            },
            'vulnerability_hotspots': sorted(
                [(facility_id, data['overall_vulnerability']) for facility_id, data in self.vulnerability_assessment.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
    
    def _count_affected_trade_corridors(self) -> int:
        """Count number of trade corridors affected by geopolitical constraints."""
        affected_pairs = set()
        
        for constraint in self.geopolitical_constraints.values():
            if constraint.expiration_date is None or constraint.expiration_date > datetime.now():
                for origin_country in constraint.origin_countries:
                    for dest_country in constraint.destination_countries:
                        affected_pairs.add((origin_country, dest_country))
        
        return len(affected_pairs)
