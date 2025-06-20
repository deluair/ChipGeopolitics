"""
Visualization Engine for ChipGeopolitics Simulation Framework

Comprehensive visualization and dashboard system for semiconductor industry
geopolitical simulation results and analytics.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class ChartType(Enum):
    """Types of charts and visualizations."""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    SANKEY_DIAGRAM = "sankey_diagram"
    NETWORK_GRAPH = "network_graph"
    GEOGRAPHIC_MAP = "geographic_map"
    TIME_SERIES = "time_series"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    RADAR_CHART = "radar_chart"

class DashboardLayout(Enum):
    """Dashboard layout types."""
    GRID = "grid"
    TABS = "tabs"
    SIDEBAR = "sidebar"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    CUSTOM = "custom"

class InteractionType(Enum):
    """Types of chart interactions."""
    HOVER = "hover"
    CLICK = "click"
    ZOOM = "zoom"
    PAN = "pan"
    BRUSH = "brush"
    FILTER = "filter"
    DRILL_DOWN = "drill_down"

@dataclass
class ChartConfiguration:
    """Configuration for individual charts."""
    chart_id: str
    chart_type: ChartType
    title: str
    data_source: str
    x_axis: Dict[str, Any]
    y_axis: Dict[str, Any]
    color_scheme: str
    dimensions: Dict[str, int]  # width, height
    interactions: List[InteractionType]
    filters: Dict[str, Any]
    styling: Dict[str, Any]

@dataclass
class DashboardConfiguration:
    """Configuration for dashboards."""
    dashboard_id: str
    title: str
    description: str
    layout: DashboardLayout
    charts: List[str]  # Chart IDs
    global_filters: Dict[str, Any]
    refresh_interval: Optional[int]  # Seconds
    export_options: List[str]
    access_permissions: List[str]

@dataclass
class VisualizationTheme:
    """Theme configuration for visualizations."""
    theme_name: str
    primary_colors: List[str]
    background_color: str
    text_color: str
    grid_color: str
    font_family: str
    font_sizes: Dict[str, int]
    chart_styles: Dict[str, Any]

class VisualizationEngine:
    """
    Comprehensive visualization and dashboard system.
    
    Capabilities:
    - Interactive chart generation
    - Dashboard creation and management
    - Real-time data visualization
    - Export and sharing functionality
    - Theme and styling customization
    - Multi-format output support
    - Performance optimization
    """
    
    def __init__(self):
        # Configuration and themes
        self.chart_configs: Dict[str, ChartConfiguration] = {}
        self.dashboard_configs: Dict[str, DashboardConfiguration] = {}
        self.themes: Dict[str, VisualizationTheme] = {}
        
        # Data and visualizations
        self.data_sources: Dict[str, Any] = {}
        self.generated_charts: Dict[str, Any] = {}
        self.dashboard_instances: Dict[str, Any] = {}
        
        # Initialize default configurations
        self._initialize_themes()
        self._initialize_chart_templates()
        self._initialize_dashboard_templates()
    
    def _initialize_themes(self):
        """Initialize visualization themes."""
        
        # Professional theme
        self.themes["professional"] = VisualizationTheme(
            theme_name="professional",
            primary_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
            background_color="#ffffff",
            text_color="#333333",
            grid_color="#e6e6e6",
            font_family="Arial, sans-serif",
            font_sizes={"title": 16, "axis": 12, "legend": 11, "tooltip": 10},
            chart_styles={
                "line_width": 2,
                "marker_size": 6,
                "opacity": 0.8,
                "border_width": 1
            }
        )
        
        # Dark theme
        self.themes["dark"] = VisualizationTheme(
            theme_name="dark",
            primary_colors=["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
                          "#34495e", "#e67e22", "#95a5a6", "#1abc9c", "#f1c40f"],
            background_color="#2c3e50",
            text_color="#ecf0f1",
            grid_color="#34495e",
            font_family="Arial, sans-serif",
            font_sizes={"title": 16, "axis": 12, "legend": 11, "tooltip": 10},
            chart_styles={
                "line_width": 2,
                "marker_size": 6,
                "opacity": 0.9,
                "border_width": 1
            }
        )
        
        # Semiconductor industry theme
        self.themes["semiconductor"] = VisualizationTheme(
            theme_name="semiconductor",
            primary_colors=["#0066cc", "#ff6600", "#00cc66", "#cc0066", "#6600cc",
                          "#cc6600", "#0099cc", "#cc9900", "#9900cc", "#00cccc"],
            background_color="#f8f9fa",
            text_color="#2c3e50",
            grid_color="#dee2e6",
            font_family="Roboto, sans-serif",
            font_sizes={"title": 18, "axis": 13, "legend": 12, "tooltip": 11},
            chart_styles={
                "line_width": 3,
                "marker_size": 8,
                "opacity": 0.85,
                "border_width": 2
            }
        )
    
    def _initialize_chart_templates(self):
        """Initialize standard chart templates."""
        
        # Market performance overview
        self.chart_configs["market_overview"] = ChartConfiguration(
            chart_id="market_overview",
            chart_type=ChartType.LINE_CHART,
            title="Semiconductor Market Growth Trends",
            data_source="market_data",
            x_axis={"field": "year", "title": "Year", "type": "temporal"},
            y_axis={"field": "market_size", "title": "Market Size (Billion USD)", "type": "quantitative"},
            color_scheme="professional",
            dimensions={"width": 800, "height": 400},
            interactions=[InteractionType.HOVER, InteractionType.ZOOM],
            filters={"region": "all", "technology_node": "all"},
            styling={"show_grid": True, "show_legend": True}
        )
        
        # Supply chain network visualization
        self.chart_configs["supply_chain_network"] = ChartConfiguration(
            chart_id="supply_chain_network",
            chart_type=ChartType.NETWORK_GRAPH,
            title="Global Semiconductor Supply Chain Network",
            data_source="supply_chain_data",
            x_axis={"field": "x_position", "title": "", "type": "quantitative"},
            y_axis={"field": "y_position", "title": "", "type": "quantitative"},
            color_scheme="semiconductor",
            dimensions={"width": 1000, "height": 600},
            interactions=[InteractionType.HOVER, InteractionType.CLICK, InteractionType.DRILL_DOWN],
            filters={"node_type": "all", "risk_level": "all"},
            styling={"node_size_field": "importance", "edge_width_field": "flow_volume"}
        )
        
        # Geopolitical risk heatmap
        self.chart_configs["risk_heatmap"] = ChartConfiguration(
            chart_id="risk_heatmap",
            chart_type=ChartType.HEATMAP,
            title="Geopolitical Risk Assessment by Region and Technology",
            data_source="geopolitical_data",
            x_axis={"field": "region", "title": "Region", "type": "nominal"},
            y_axis={"field": "technology", "title": "Technology Area", "type": "nominal"},
            color_scheme="professional",
            dimensions={"width": 600, "height": 400},
            interactions=[InteractionType.HOVER, InteractionType.CLICK],
            filters={"year": 2024, "scenario": "baseline"},
            styling={"color_field": "risk_score", "text_overlay": True}
        )
        
        # Company performance comparison
        self.chart_configs["company_comparison"] = ChartConfiguration(
            chart_id="company_comparison",
            chart_type=ChartType.RADAR_CHART,
            title="Company Performance Comparison",
            data_source="company_data",
            x_axis={"field": "metric", "title": "Performance Metrics", "type": "nominal"},
            y_axis={"field": "score", "title": "Score", "type": "quantitative"},
            color_scheme="semiconductor",
            dimensions={"width": 500, "height": 500},
            interactions=[InteractionType.HOVER, InteractionType.FILTER],
            filters={"companies": ["tsmc", "samsung", "intel"], "year": 2024},
            styling={"fill_opacity": 0.3, "stroke_width": 2}
        )
        
        # Energy consumption trends
        self.chart_configs["energy_trends"] = ChartConfiguration(
            chart_id="energy_trends",
            chart_type=ChartType.BAR_CHART,
            title="Energy Consumption by Manufacturing Node",
            data_source="energy_data",
            x_axis={"field": "process_node", "title": "Process Node (nm)", "type": "ordinal"},
            y_axis={"field": "energy_intensity", "title": "Energy Intensity (kWh/wafer)", "type": "quantitative"},
            color_scheme="professional",
            dimensions={"width": 700, "height": 400},
            interactions=[InteractionType.HOVER, InteractionType.CLICK],
            filters={"year": 2024, "company": "all"},
            styling={"show_values": True, "gradient_fill": True}
        )
    
    def _initialize_dashboard_templates(self):
        """Initialize standard dashboard templates."""
        
        # Executive overview dashboard
        self.dashboard_configs["executive_overview"] = DashboardConfiguration(
            dashboard_id="executive_overview",
            title="Executive Overview Dashboard",
            description="High-level view of semiconductor industry trends and risks",
            layout=DashboardLayout.GRID,
            charts=["market_overview", "risk_heatmap", "company_comparison"],
            global_filters={"year": 2024, "scenario": "baseline"},
            refresh_interval=300,  # 5 minutes
            export_options=["pdf", "png", "csv"],
            access_permissions=["executive", "analyst"]
        )
        
        # Supply chain dashboard
        self.dashboard_configs["supply_chain"] = DashboardConfiguration(
            dashboard_id="supply_chain",
            title="Supply Chain Analysis Dashboard",
            description="Comprehensive supply chain network and risk analysis",
            layout=DashboardLayout.TABS,
            charts=["supply_chain_network", "risk_heatmap", "energy_trends"],
            global_filters={"scenario": "baseline", "risk_threshold": 0.7},
            refresh_interval=600,  # 10 minutes
            export_options=["pdf", "png"],
            access_permissions=["supply_chain", "analyst", "executive"]
        )
        
        # Market analysis dashboard
        self.dashboard_configs["market_analysis"] = DashboardConfiguration(
            dashboard_id="market_analysis",
            title="Market Dynamics Dashboard",
            description="Market trends, competitive analysis, and forecasting",
            layout=DashboardLayout.VERTICAL,
            charts=["market_overview", "company_comparison", "energy_trends"],
            global_filters={"region": "global", "technology_focus": "advanced_nodes"},
            refresh_interval=1800,  # 30 minutes
            export_options=["pdf", "excel", "png"],
            access_permissions=["market_analyst", "executive", "sales"]
        )
    
    def create_chart(self, chart_config: ChartConfiguration, 
                    data: Any, theme: str = "professional") -> Dict[str, Any]:
        """Create a chart based on configuration and data."""
        
        if theme not in self.themes:
            theme = "professional"
        
        chart_theme = self.themes[theme]
        
        # Generate chart specification based on type
        if chart_config.chart_type == ChartType.LINE_CHART:
            chart_spec = self._create_line_chart(chart_config, data, chart_theme)
        elif chart_config.chart_type == ChartType.BAR_CHART:
            chart_spec = self._create_bar_chart(chart_config, data, chart_theme)
        elif chart_config.chart_type == ChartType.SCATTER_PLOT:
            chart_spec = self._create_scatter_plot(chart_config, data, chart_theme)
        elif chart_config.chart_type == ChartType.HEATMAP:
            chart_spec = self._create_heatmap(chart_config, data, chart_theme)
        elif chart_config.chart_type == ChartType.NETWORK_GRAPH:
            chart_spec = self._create_network_graph(chart_config, data, chart_theme)
        elif chart_config.chart_type == ChartType.RADAR_CHART:
            chart_spec = self._create_radar_chart(chart_config, data, chart_theme)
        else:
            chart_spec = self._create_default_chart(chart_config, data, chart_theme)
        
        # Store generated chart
        self.generated_charts[chart_config.chart_id] = chart_spec
        
        return chart_spec
    
    def _create_line_chart(self, config: ChartConfiguration, 
                          data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create line chart specification."""
        
        return {
            "chart_type": "line",
            "title": config.title,
            "data": data,
            "encoding": {
                "x": {
                    "field": config.x_axis["field"],
                    "type": config.x_axis["type"],
                    "title": config.x_axis["title"],
                    "axis": {"grid": config.styling.get("show_grid", True)}
                },
                "y": {
                    "field": config.y_axis["field"],
                    "type": config.y_axis["type"],
                    "title": config.y_axis["title"],
                    "axis": {"grid": config.styling.get("show_grid", True)}
                },
                "color": {
                    "field": config.filters.get("color_field", "series"),
                    "scale": {"range": theme.primary_colors}
                }
            },
            "mark": {
                "type": "line",
                "strokeWidth": theme.chart_styles["line_width"],
                "opacity": theme.chart_styles["opacity"],
                "point": {
                    "size": theme.chart_styles["marker_size"] ** 2,
                    "filled": True
                }
            },
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color,
            "config": {
                "title": {"fontSize": theme.font_sizes["title"], "color": theme.text_color},
                "axis": {"labelFontSize": theme.font_sizes["axis"], "titleFontSize": theme.font_sizes["axis"]},
                "legend": {"labelFontSize": theme.font_sizes["legend"]}
            },
            "selection": self._create_interactions(config.interactions),
            "transform": self._apply_filters(config.filters)
        }
    
    def _create_bar_chart(self, config: ChartConfiguration, 
                         data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create bar chart specification."""
        
        return {
            "chart_type": "bar",
            "title": config.title,
            "data": data,
            "encoding": {
                "x": {
                    "field": config.x_axis["field"],
                    "type": config.x_axis["type"],
                    "title": config.x_axis["title"]
                },
                "y": {
                    "field": config.y_axis["field"],
                    "type": config.y_axis["type"],
                    "title": config.y_axis["title"]
                },
                "color": {
                    "field": config.filters.get("color_field", "category"),
                    "scale": {"range": theme.primary_colors}
                }
            },
            "mark": {
                "type": "bar",
                "opacity": theme.chart_styles["opacity"],
                "strokeWidth": theme.chart_styles["border_width"]
            },
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color,
            "config": {
                "title": {"fontSize": theme.font_sizes["title"], "color": theme.text_color},
                "axis": {"labelFontSize": theme.font_sizes["axis"], "titleFontSize": theme.font_sizes["axis"]}
            },
            "selection": self._create_interactions(config.interactions),
            "transform": self._apply_filters(config.filters)
        }
    
    def _create_scatter_plot(self, config: ChartConfiguration, 
                           data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create scatter plot specification."""
        
        return {
            "chart_type": "scatter",
            "title": config.title,
            "data": data,
            "encoding": {
                "x": {
                    "field": config.x_axis["field"],
                    "type": config.x_axis["type"],
                    "title": config.x_axis["title"]
                },
                "y": {
                    "field": config.y_axis["field"],
                    "type": config.y_axis["type"],
                    "title": config.y_axis["title"]
                },
                "color": {
                    "field": config.filters.get("color_field", "group"),
                    "scale": {"range": theme.primary_colors}
                },
                "size": {
                    "field": config.styling.get("size_field", "value"),
                    "scale": {"range": [50, 400]}
                }
            },
            "mark": {
                "type": "circle",
                "opacity": theme.chart_styles["opacity"],
                "strokeWidth": theme.chart_styles["border_width"]
            },
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color,
            "selection": self._create_interactions(config.interactions),
            "transform": self._apply_filters(config.filters)
        }
    
    def _create_heatmap(self, config: ChartConfiguration, 
                       data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create heatmap specification."""
        
        return {
            "chart_type": "heatmap",
            "title": config.title,
            "data": data,
            "encoding": {
                "x": {
                    "field": config.x_axis["field"],
                    "type": config.x_axis["type"],
                    "title": config.x_axis["title"]
                },
                "y": {
                    "field": config.y_axis["field"],
                    "type": config.y_axis["type"],
                    "title": config.y_axis["title"]
                },
                "color": {
                    "field": config.styling.get("color_field", "value"),
                    "type": "quantitative",
                    "scale": {"scheme": "blues"}
                }
            },
            "mark": {
                "type": "rect",
                "strokeWidth": 1,
                "stroke": theme.grid_color
            },
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color,
            "selection": self._create_interactions(config.interactions),
            "transform": self._apply_filters(config.filters)
        }
    
    def _create_network_graph(self, config: ChartConfiguration, 
                            data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create network graph specification."""
        
        return {
            "chart_type": "network",
            "title": config.title,
            "data": {
                "nodes": data.get("nodes", []),
                "edges": data.get("edges", [])
            },
            "layout": {
                "type": "force_directed",
                "iterations": 100,
                "node_strength": -300,
                "link_distance": 30
            },
            "node_encoding": {
                "size": {
                    "field": config.styling.get("node_size_field", "degree"),
                    "scale": {"range": [50, 500]}
                },
                "color": {
                    "field": config.filters.get("color_field", "type"),
                    "scale": {"range": theme.primary_colors}
                }
            },
            "edge_encoding": {
                "strokeWidth": {
                    "field": config.styling.get("edge_width_field", "weight"),
                    "scale": {"range": [1, 8]}
                },
                "stroke": theme.grid_color,
                "opacity": 0.6
            },
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color,
            "interactions": config.interactions
        }
    
    def _create_radar_chart(self, config: ChartConfiguration, 
                          data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create radar chart specification."""
        
        return {
            "chart_type": "radar",
            "title": config.title,
            "data": data,
            "encoding": {
                "theta": {
                    "field": config.x_axis["field"],
                    "type": config.x_axis["type"],
                    "scale": {"range": [0, 6.28]}  # 2 * pi
                },
                "radius": {
                    "field": config.y_axis["field"],
                    "type": config.y_axis["type"],
                    "scale": {"range": [0, 100]}
                },
                "color": {
                    "field": config.filters.get("color_field", "series"),
                    "scale": {"range": theme.primary_colors}
                }
            },
            "mark": {
                "type": "area",
                "opacity": config.styling.get("fill_opacity", 0.3),
                "strokeWidth": config.styling.get("stroke_width", 2)
            },
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color,
            "selection": self._create_interactions(config.interactions)
        }
    
    def _create_default_chart(self, config: ChartConfiguration, 
                            data: Any, theme: VisualizationTheme) -> Dict[str, Any]:
        """Create default chart when specific type not implemented."""
        
        return {
            "chart_type": "default",
            "title": config.title,
            "data": data,
            "message": f"Chart type {config.chart_type.value} not yet implemented",
            "width": config.dimensions["width"],
            "height": config.dimensions["height"],
            "background": theme.background_color
        }
    
    def _create_interactions(self, interactions: List[InteractionType]) -> Dict[str, Any]:
        """Create interaction specifications for charts."""
        
        interaction_spec = {}
        
        for interaction in interactions:
            if interaction == InteractionType.HOVER:
                interaction_spec["hover"] = {
                    "type": "single",
                    "on": "mouseover",
                    "clear": "mouseout"
                }
            elif interaction == InteractionType.CLICK:
                interaction_spec["click"] = {
                    "type": "single",
                    "on": "click"
                }
            elif interaction == InteractionType.ZOOM:
                interaction_spec["zoom"] = {
                    "type": "interval",
                    "bind": "scales"
                }
            elif interaction == InteractionType.BRUSH:
                interaction_spec["brush"] = {
                    "type": "interval"
                }
        
        return interaction_spec
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply data filters to chart."""
        
        transforms = []
        
        for field, value in filters.items():
            if value != "all" and value is not None:
                transforms.append({
                    "filter": {
                        "field": field,
                        "equal": value
                    }
                })
        
        return transforms
    
    def create_dashboard(self, dashboard_config: DashboardConfiguration, 
                        theme: str = "professional") -> Dict[str, Any]:
        """Create dashboard with multiple charts."""
        
        dashboard_spec = {
            "dashboard_id": dashboard_config.dashboard_id,
            "title": dashboard_config.title,
            "description": dashboard_config.description,
            "layout": dashboard_config.layout.value,
            "theme": theme,
            "charts": [],
            "global_filters": dashboard_config.global_filters,
            "refresh_interval": dashboard_config.refresh_interval,
            "export_options": dashboard_config.export_options,
            "created_at": datetime.now().isoformat()
        }
        
        # Add chart specifications
        for chart_id in dashboard_config.charts:
            if chart_id in self.chart_configs:
                chart_config = self.chart_configs[chart_id]
                
                # Apply global filters to chart
                updated_filters = {**chart_config.filters, **dashboard_config.global_filters}
                chart_config.filters = updated_filters
                
                # Get or generate chart data
                chart_data = self._get_chart_data(chart_config.data_source)
                
                # Create chart
                chart_spec = self.create_chart(chart_config, chart_data, theme)
                dashboard_spec["charts"].append(chart_spec)
        
        # Apply layout-specific configurations
        if dashboard_config.layout == DashboardLayout.GRID:
            dashboard_spec["grid_config"] = self._create_grid_layout(dashboard_config.charts)
        elif dashboard_config.layout == DashboardLayout.TABS:
            dashboard_spec["tab_config"] = self._create_tab_layout(dashboard_config.charts)
        
        self.dashboard_instances[dashboard_config.dashboard_id] = dashboard_spec
        return dashboard_spec
    
    def _get_chart_data(self, data_source: str) -> Any:
        """Get data for chart from data source."""
        
        # Placeholder for actual data retrieval
        # In real implementation, this would connect to simulation results
        
        if data_source == "market_data":
            return self._generate_sample_market_data()
        elif data_source == "supply_chain_data":
            return self._generate_sample_supply_chain_data()
        elif data_source == "geopolitical_data":
            return self._generate_sample_geopolitical_data()
        elif data_source == "company_data":
            return self._generate_sample_company_data()
        elif data_source == "energy_data":
            return self._generate_sample_energy_data()
        else:
            return []
    
    def _generate_sample_market_data(self) -> List[Dict[str, Any]]:
        """Generate sample market data for demonstration."""
        
        data = []
        for year in range(2020, 2031):
            market_size = 450 + (year - 2020) * 35 + np.random.normal(0, 20)
            data.append({
                "year": year,
                "market_size": max(400, market_size),
                "growth_rate": 0.06 + np.random.normal(0, 0.02),
                "region": "Global"
            })
        
        return data
    
    def _generate_sample_supply_chain_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate sample supply chain network data."""
        
        nodes = [
            {"id": "tsmc", "name": "TSMC", "type": "foundry", "region": "Taiwan", 
             "importance": 95, "x_position": 120, "y_position": 80},
            {"id": "samsung", "name": "Samsung", "type": "foundry", "region": "South Korea", 
             "importance": 85, "x_position": 140, "y_position": 70},
            {"id": "intel", "name": "Intel", "type": "foundry", "region": "USA", 
             "importance": 75, "x_position": 50, "y_position": 90},
            {"id": "asml", "name": "ASML", "type": "equipment", "region": "Netherlands", 
             "importance": 90, "x_position": 80, "y_position": 60},
            {"id": "applied_materials", "name": "Applied Materials", "type": "equipment", 
             "region": "USA", "importance": 70, "x_position": 60, "y_position": 50}
        ]
        
        edges = [
            {"source": "asml", "target": "tsmc", "flow_volume": 85, "relationship": "supplier"},
            {"source": "asml", "target": "samsung", "flow_volume": 70, "relationship": "supplier"},
            {"source": "applied_materials", "target": "tsmc", "flow_volume": 60, "relationship": "supplier"},
            {"source": "applied_materials", "target": "intel", "flow_volume": 80, "relationship": "supplier"}
        ]
        
        return {"nodes": nodes, "edges": edges}
    
    def _generate_sample_geopolitical_data(self) -> List[Dict[str, Any]]:
        """Generate sample geopolitical risk data."""
        
        regions = ["USA", "China", "Taiwan", "South Korea", "Japan", "EU"]
        technologies = ["Advanced Logic", "Memory", "Analog", "Equipment", "Materials"]
        
        data = []
        for region in regions:
            for tech in technologies:
                risk_score = np.random.uniform(0.2, 0.9)
                data.append({
                    "region": region,
                    "technology": tech,
                    "risk_score": risk_score,
                    "year": 2024,
                    "scenario": "baseline"
                })
        
        return data
    
    def _generate_sample_company_data(self) -> List[Dict[str, Any]]:
        """Generate sample company performance data."""
        
        companies = ["TSMC", "Samsung", "Intel"]
        metrics = ["Innovation", "Market Share", "Financial Health", "Supply Chain", "ESG Score"]
        
        data = []
        for company in companies:
            for metric in metrics:
                score = np.random.uniform(60, 95)
                data.append({
                    "company": company,
                    "metric": metric,
                    "score": score,
                    "year": 2024
                })
        
        return data
    
    def _generate_sample_energy_data(self) -> List[Dict[str, Any]]:
        """Generate sample energy consumption data."""
        
        process_nodes = ["3nm", "5nm", "7nm", "14nm", "28nm", "65nm"]
        
        data = []
        for node in process_nodes:
            # More advanced nodes consume more energy
            base_energy = 50 + (180 - int(node.replace("nm", ""))) * 2
            energy_intensity = base_energy + np.random.normal(0, 10)
            
            data.append({
                "process_node": node,
                "energy_intensity": max(20, energy_intensity),
                "year": 2024,
                "company": "Industry Average"
            })
        
        return data
    
    def _create_grid_layout(self, chart_ids: List[str]) -> Dict[str, Any]:
        """Create grid layout configuration."""
        
        num_charts = len(chart_ids)
        if num_charts <= 2:
            cols = num_charts
            rows = 1
        elif num_charts <= 4:
            cols = 2
            rows = 2
        else:
            cols = 3
            rows = (num_charts + 2) // 3
        
        return {
            "type": "grid",
            "columns": cols,
            "rows": rows,
            "spacing": {"horizontal": 20, "vertical": 20},
            "responsive": True
        }
    
    def _create_tab_layout(self, chart_ids: List[str]) -> Dict[str, Any]:
        """Create tab layout configuration."""
        
        return {
            "type": "tabs",
            "orientation": "horizontal",
            "tab_position": "top",
            "tabs": [{"id": chart_id, "title": chart_id.replace("_", " ").title()} 
                    for chart_id in chart_ids]
        }
    
    def export_chart(self, chart_id: str, format: str = "png", 
                    options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export chart to specified format."""
        
        if chart_id not in self.generated_charts:
            return {"error": "Chart not found"}
        
        chart_spec = self.generated_charts[chart_id]
        export_options = options or {}
        
        export_result = {
            "chart_id": chart_id,
            "format": format,
            "exported_at": datetime.now().isoformat(),
            "file_path": f"exports/{chart_id}.{format}",
            "options": export_options
        }
        
        # Placeholder for actual export implementation
        if format == "png":
            export_result["width"] = export_options.get("width", chart_spec.get("width", 800))
            export_result["height"] = export_options.get("height", chart_spec.get("height", 600))
            export_result["dpi"] = export_options.get("dpi", 300)
        elif format == "pdf":
            export_result["page_size"] = export_options.get("page_size", "A4")
            export_result["orientation"] = export_options.get("orientation", "landscape")
        elif format == "svg":
            export_result["vector_format"] = True
            export_result["scalable"] = True
        
        return export_result
    
    def export_dashboard(self, dashboard_id: str, format: str = "pdf", 
                        options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Export entire dashboard to specified format."""
        
        if dashboard_id not in self.dashboard_instances:
            return {"error": "Dashboard not found"}
        
        dashboard_spec = self.dashboard_instances[dashboard_id]
        export_options = options or {}
        
        export_result = {
            "dashboard_id": dashboard_id,
            "format": format,
            "exported_at": datetime.now().isoformat(),
            "file_path": f"exports/{dashboard_id}.{format}",
            "options": export_options,
            "charts_included": len(dashboard_spec["charts"])
        }
        
        # Format-specific options
        if format == "pdf":
            export_result["multi_page"] = export_options.get("multi_page", True)
            export_result["include_data"] = export_options.get("include_data", False)
        elif format == "html":
            export_result["interactive"] = export_options.get("interactive", True)
            export_result["standalone"] = export_options.get("standalone", True)
        
        return export_result
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of visualization capabilities."""
        
        return {
            "chart_types_supported": [chart_type.value for chart_type in ChartType],
            "themes_available": list(self.themes.keys()),
            "chart_templates": len(self.chart_configs),
            "dashboard_templates": len(self.dashboard_configs),
            "generated_charts": len(self.generated_charts),
            "active_dashboards": len(self.dashboard_instances),
            "export_formats": ["png", "pdf", "svg", "html", "csv", "json"],
            "interaction_types": [interaction.value for interaction in InteractionType],
            "layout_options": [layout.value for layout in DashboardLayout],
            "capabilities": [
                "Interactive chart generation",
                "Multi-chart dashboard creation",
                "Real-time data visualization",
                "Custom theming and styling",
                "Export to multiple formats",
                "Responsive design support",
                "Advanced interaction handling"
            ]
        }