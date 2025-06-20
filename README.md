# ChipGeopolitics Simulation Framework

A comprehensive agent-based simulation framework for modeling the complex interdependencies between AI data center expansion, semiconductor supply chains, energy infrastructure, and geopolitical tensions in the period 2025-2035.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Mesa](https://img.shields.io/badge/mesa-1.1.1-green.svg)](https://mesa.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

This simulation framework models the strategic interactions between hyperscalers (cloud providers), semiconductor manufacturers, equipment suppliers, and nation-states as they navigate an increasingly complex geopolitical landscape shaped by AI demand, supply chain vulnerabilities, and international tensions.

### 🎯 Key Features

- **✅ WORKING SIMULATION**: All core components tested and functional
- **📊 Industry-Accurate Data**: Based on real 2024 semiconductor industry metrics
- **🎲 Advanced Monte Carlo**: Full uncertainty analysis with multiple distributions
- **🏭 Realistic Company Profiles**: 500+ synthetic companies with accurate characteristics
- **🌍 Geopolitical Modeling**: Export controls, trade tensions, supply chain disruptions
- **⚡ Performance Optimized**: Parallel execution and efficient algorithms

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git

### Installation & Demo
```bash
# Clone the repository
git clone https://github.com/deluair/ChipGeopolitics.git
cd ChipGeopolitics

# Install dependencies
pip install mesa numpy pandas scipy networkx

# Test all components
python test_components.py

# Run comprehensive demonstration
python demo_simulation.py
```

## 📋 Demonstration Results

The framework successfully demonstrates:

### 🏭 Realistic Data Generation
```
✅ Generated 10 Hyperscaler profiles
✅ Generated 20 Chip Manufacturer profiles  
✅ Generated 15 Equipment Supplier profiles
✅ Generated 8 Nation-State profiles
```

**Sample Hyperscaler Profile:**
- Annual CAPEX: $25.6B USD
- Data Centers: 71 facilities
- Chip Procurement: 358,795 units/year
- Cloud Market Share: 5.2%
- Global Presence: 4 regions

### 🎲 Monte Carlo Analysis
```
✅ Created 4 realistic scenarios:
  • baseline (40%): Current trends continue
  • trade_war_escalation (30%): US-China tensions escalate
  • quantum_breakthrough (20%): Major tech breakthrough
  • major_supply_crisis (10%): Taiwan earthquake/disruption
```

### 📊 Industry Baseline Data (2024)
- **AI Data Centers**: 415 TWh → 945 TWh (2030), $455B CAPEX
- **Hyperscaler Investments**: Microsoft $80B, Google $75B, Meta $65B, Amazon $60B
- **TSMC Market Share**: 61.7% (projected 75% by 2026)
- **Process Node Costs**: 28nm ($2,500) to 5nm ($9,566) per wafer
- **China Self-Sufficiency**: 24% (2020) → 80% target (2030)

## 🏗️ Architecture

### Core Components

#### 1. **Agent Framework** (`src/core/base_agent.py`)
```python
class BaseAgent(Agent):
    """Sophisticated agent with economic and strategic capabilities"""
    - Economic attributes (revenue, costs, profit margins)
    - Supply chain relationships and dependencies  
    - Geopolitical risk assessment
    - Strategic decision making framework
```

#### 2. **Monte Carlo Engine** (`src/core/monte_carlo.py`)
```python
# Support for 7 distribution types
distributions = [Normal, Uniform, Beta, Gamma, Lognormal, Exponential, Triangular]

# Risk metrics
risk_metrics = ["VaR_95", "VaR_99", "Expected_Shortfall", "Volatility"]

# Scenario-based analysis with probability weighting
```

#### 3. **Data Generation** (`src/data_generation/company_profiles.py`)
```python
# Generate realistic company profiles
generator = CompanyProfileGenerator(random_seed=42)
hyperscalers = generator.generate_hyperscaler_profiles(10)
chip_manufacturers = generator.generate_chip_manufacturer_profiles(50)
```

#### 4. **Configuration Management** (`config/`)
- **Industry Constants**: Real 2024 baseline data and projections
- **Geopolitical Timeline**: Export control escalation dates
- **Economic Parameters**: CAPEX, market shares, investment flows

## 📖 Usage Examples

### Basic Component Testing
```python
# Test all core components
python test_components.py
```
**Output:**
```
✅ CAPEX 2024: $455B USD
✅ Generated 3 hyperscaler profiles  
✅ Monte Carlo engine created
✅ Base agent created with strategic capabilities
🎉 All core components are working!
```

### Monte Carlo Scenario Analysis
```python
from src.core.monte_carlo import MonteCarloEngine, ScenarioDefinition

monte_carlo = MonteCarloEngine(random_seed=42)

# Define geopolitical scenarios
baseline_scenario = ScenarioDefinition(
    name="baseline",
    description="Current trends continue",
    variable_distributions={
        "chip_demand_growth": DistributionParams(DistributionType.NORMAL, {"mean": 0.30, "std": 0.05}),
        "geopolitical_tension": DistributionParams(DistributionType.BETA, {"alpha": 2, "beta": 3})
    },
    probability=0.4
)

results = monte_carlo.run_simulation(model_function, n_iterations=500)
```

### Company Profile Generation
```python
from src.data_generation.company_profiles import CompanyProfileGenerator

generator = CompanyProfileGenerator(random_seed=42)

# Generate realistic companies
hyperscalers = generator.generate_hyperscaler_profiles(10)
print(f"Generated {len(hyperscalers)} hyperscaler profiles")

# Sample profile data
for profile in hyperscalers[:2]:
    print(f"{profile.name}: ${profile.annual_capex:.1f}B CAPEX, {profile.data_center_count} DCs")
```

## 🛠️ Implementation Status

### ✅ **Phase 1: Core Infrastructure** (COMPLETED ✅)
- [x] **Project Structure**: Modular design with proper separation of concerns
- [x] **Industry Data**: Real 2024 semiconductor industry baseline 
- [x] **Monte Carlo Framework**: Advanced uncertainty analysis with 7 distributions
- [x] **Agent Architecture**: Economic modeling, supply chains, geopolitical risk
- [x] **Data Generation**: 500+ realistic company profiles across all agent types
- [x] **Testing Framework**: Comprehensive validation of all components
- [x] **Documentation**: Detailed README with usage examples

**🎉 ALL CORE COMPONENTS TESTED AND WORKING! 🎉**

### 🔄 **Phase 2: Agent Behaviors** (Next Priority)
- [ ] Hyperscaler strategic decision models (capacity planning, supplier diversification)
- [ ] Chip manufacturer technology roadmaps and capacity allocation
- [ ] Equipment supplier innovation cycles and market positioning
- [ ] Nation-state policy implementation and trade relationship dynamics

### 📈 **Phase 3: Market Dynamics** (Future Development)
- [ ] Supply-demand matching and dynamic pricing mechanisms
- [ ] Technology diffusion networks and innovation spillovers
- [ ] Geopolitical event triggers and cascading effects
- [ ] Energy infrastructure constraints and sustainability metrics

### 🔬 **Phase 4: Advanced Analytics** (Future Enhancement)
- [ ] Interactive visualization dashboard with Plotly/Dash
- [ ] Real-time data integration and model calibration
- [ ] Machine learning pattern recognition and prediction
- [ ] Policy optimization and scenario testing tools

## 💻 System Requirements

### **Minimum Requirements**
- **Python**: 3.9 or higher
- **Memory**: 4GB RAM (8GB+ recommended for large simulations)
- **CPU**: Multi-core processor recommended for parallel execution
- **Storage**: 1GB for basic installation, 2GB+ for analysis outputs

### **Dependencies**
```python
# Core simulation framework
mesa==1.1.1              # Agent-based modeling
numpy>=1.21.0            # Numerical computing  
pandas>=1.3.0            # Data manipulation
scipy>=1.7.0             # Statistical distributions

# Network modeling
networkx>=2.6.0          # Supply chain networks

# Future enhancements
tensorflow>=2.8.0        # Machine learning (optional)
plotly>=5.0.0           # Interactive visualization (optional)
```

## 🔬 Research Applications

This framework enables cutting-edge research in:

### **🛡️ Economic Security**
- Supply chain resilience under geopolitical stress
- Critical technology dependency analysis
- Strategic stockpiling and alternative supplier development

### **📜 Technology Policy** 
- Export control impact assessment on global innovation
- Technology transfer restrictions and competitive dynamics
- National security implications of AI infrastructure dependencies

### **🌱 Sustainable Computing**
- Energy consumption and carbon footprint modeling
- Renewable energy integration in data center expansion
- Environmental impact of semiconductor manufacturing shifts

### **🗺️ Geopolitical Risk Analysis**
- Scenario planning for US-China technology competition
- Regional bloc formation and technology spheres
- Critical mineral supply chain vulnerabilities

## 📁 Project Structure

```
ChipGeopolitics/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── test_components.py          # Component validation
├── demo_simulation.py          # Comprehensive demonstration
│
├── config/                     # Configuration management
│   ├── constants.py           # Industry baseline data (2024)
│   └── settings.py            # Simulation parameters
│
├── src/                       # Core simulation framework
│   ├── core/                  # Agent and simulation engine
│   │   ├── base_agent.py     # Abstract agent class
│   │   └── monte_carlo.py    # Uncertainty analysis engine
│   │
│   └── data_generation/       # Synthetic data creation
│       └── company_profiles.py # Realistic company generation
│
├── data/                      # Data storage
│   └── demo/                 # Demonstration outputs
│
├── scripts/                  # Execution scripts
│   └── run_simulation.py    # Main simulation runner
│
└── logs/                     # Task tracking and logs
    └── task_log.md          # Implementation progress
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Test** your changes (`python test_components.py`)
4. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
5. **Push** to the branch (`git push origin feature/AmazingFeature`)
6. **Open** a Pull Request

### Development Guidelines
- Follow existing code style and structure
- Add tests for new functionality
- Update documentation for significant changes
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mesa Framework**: Agent-based modeling foundation
- **Semiconductor Industry Reports**: Real industry data sources
- **Export Control Updates**: Public policy announcements and timelines
- **Research Community**: Academic literature on geopolitical economics

---

## 🔗 Links

- **Repository**: [https://github.com/deluair/ChipGeopolitics](https://github.com/deluair/ChipGeopolitics)
- **Issues**: [Report bugs or request features](https://github.com/deluair/ChipGeopolitics/issues)
- **Mesa Documentation**: [https://mesa.readthedocs.io/](https://mesa.readthedocs.io/)

---

**🚀 Ready to explore semiconductor geopolitics? Run `python demo_simulation.py` to get started!** 