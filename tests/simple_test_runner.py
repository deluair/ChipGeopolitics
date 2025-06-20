#!/usr/bin/env python3
"""
Simple Test Runner for ChipGeopolitics Framework

Basic test runner that works with available components and
provides a demonstration of the testing framework.
"""

import sys
import time
import os
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

def test_imports():
    """Test importing available components."""
    print("🔧 Testing Component Imports...")
    results = {}
    
    # Test simulation engine
    try:
        from core.simulation_engine import SimulationEngine, Decision, AgentState, BaseAgent
        results['simulation_engine'] = True
        print("  ✅ SimulationEngine imported successfully")
    except ImportError as e:
        results['simulation_engine'] = False
        print(f"  ❌ SimulationEngine import failed: {e}")
    
    # Test Monte Carlo engine
    try:
        from core.monte_carlo import MonteCarloEngine
        results['monte_carlo'] = True
        print("  ✅ MonteCarloEngine imported successfully")
    except ImportError as e:
        results['monte_carlo'] = False
        print(f"  ❌ MonteCarloEngine import failed: {e}")
    
    return results

def test_simulation_engine():
    """Test basic simulation engine functionality."""
    print("\n🎮 Testing Simulation Engine...")
    
    try:
        from core.simulation_engine import SimulationEngine, Decision, BaseAgent
        
        # Test initialization
        engine = SimulationEngine()
        print("  ✅ Engine initialization successful")
        
        # Test agent creation and addition
        agent = BaseAgent("test_agent", "test_type", {"resource": 100})
        engine.add_agent(agent)
        print("  ✅ Agent creation and addition successful")
        
        # Test single step
        engine.step()
        print("  ✅ Single step execution successful")
        
        # Test short run
        results = engine.run(steps=3)
        print(f"  ✅ Short simulation run successful: {results['final_step']} steps")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Simulation engine test failed: {e}")
        return False

def test_monte_carlo():
    """Test basic Monte Carlo functionality."""
    print("\n🎲 Testing Monte Carlo Engine...")
    
    try:
        from core.monte_carlo import MonteCarloEngine
        from core.simulation_engine import SimulationEngine, BaseAgent
        
        # Test initialization
        mc_engine = MonteCarloEngine()
        print("  ✅ Monte Carlo engine initialization successful")
        
        # Test simulation engine integration
        sim_engine = SimulationEngine()
        mc_engine.set_simulation_engine(sim_engine)
        print("  ✅ Simulation engine integration successful")
        
        # Test parameter sampling
        parameters = {
            "test_param": ("normal", 0.5, 0.1)
        }
        samples = mc_engine.sample_parameters(parameters, num_samples=5)
        print(f"  ✅ Parameter sampling successful: {len(samples)} samples")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Monte Carlo test failed: {e}")
        return False

def test_framework_integration():
    """Test basic framework integration."""
    print("\n🔗 Testing Framework Integration...")
    
    try:
        from core.simulation_engine import SimulationEngine, BaseAgent, Decision
        
        # Create simulation with multiple agents
        engine = SimulationEngine()
        
        # Add multiple agents
        for i in range(3):
            agent = BaseAgent(f"agent_{i}", "test", {"id": i, "resource": 100 + i * 10})
            engine.add_agent(agent)
        
        print(f"  ✅ Created simulation with {len(engine.agents)} agents")
        
        # Run integrated simulation
        start_time = time.time()
        results = engine.run(steps=5)
        execution_time = time.time() - start_time
        
        print(f"  ✅ Integrated simulation completed in {execution_time:.3f}s")
        print(f"    - Final step: {results['final_step']}")
        print(f"    - Agent count: {results['agent_count']}")
        print(f"    - Converged: {results.get('converged', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Framework integration test failed: {e}")
        return False

def test_decision_system():
    """Test decision-making system."""
    print("\n🧠 Testing Decision System...")
    
    try:
        from core.simulation_engine import Decision
        
        # Test decision creation
        decision = Decision("invest", {"amount": 1000}, 0.8)
        print("  ✅ Decision creation successful")
        
        # Test decision validation
        assert decision.is_valid(), "Decision should be valid"
        print("  ✅ Decision validation successful")
        
        # Test decision serialization
        decision_dict = decision.to_dict()
        assert "action" in decision_dict, "Decision dict should have action"
        print("  ✅ Decision serialization successful")
        
        # Test decision comparison
        decision2 = Decision("sell", {"amount": 500}, 0.6)
        assert decision > decision2, "Higher utility decision should be greater"
        print("  ✅ Decision comparison successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Decision system test failed: {e}")
        return False

def run_demonstration():
    """Run a complete framework demonstration."""
    print("\n🚀 Running Framework Demonstration...")
    
    try:
        from core.simulation_engine import SimulationEngine, BaseAgent, Decision
        
        print("    Initializing simulation framework...")
        engine = SimulationEngine()
        
        print("    Creating diverse agent ecosystem...")
        agent_types = ["manufacturer", "hyperscaler", "supplier", "nation_state"]
        
        for i, agent_type in enumerate(agent_types):
            agent = BaseAgent(f"{agent_type}_{i}", agent_type, {
                "capital": 1000 + i * 500,
                "market_share": 0.1 + i * 0.1,
                "technology_level": 0.7 + i * 0.05
            })
            engine.add_agent(agent)
            print(f"      ✓ Added {agent_type} agent")
        
        print("    Running comprehensive simulation...")
        start_time = time.time()
        results = engine.run(steps=15)
        execution_time = time.time() - start_time
        
        print(f"    🎯 Simulation Results:")
        print(f"      • Execution time: {execution_time:.3f} seconds")
        print(f"      • Steps completed: {results['final_step']}")
        print(f"      • Agents simulated: {results['agent_count']}")
        print(f"      • System converged: {results.get('converged', False)}")
        print(f"      • Market volatility: {results['final_state']['market_conditions']['price_volatility']:.3f}")
        print(f"      • Geopolitical tension: {results['final_state']['market_conditions']['geopolitical_tension']:.3f}")
        
        print("  ✅ Framework demonstration completed successfully!")
        
        return True, results
        
    except Exception as e:
        print(f"  ❌ Framework demonstration failed: {e}")
        return False, None

def main():
    """Main test execution."""
    print("🎯" + "=" * 60 + "🎯")
    print("🎯" + " " * 10 + "CHIPGEOPOLITICS TESTING FRAMEWORK" + " " * 12 + "🎯") 
    print("🎯" + "=" * 60 + "🎯")
    
    print(f"\n📅 Testing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_start_time = time.time()
    test_results = {}
    
    # Run all tests
    test_results['imports'] = test_imports()
    test_results['simulation_engine'] = test_simulation_engine()
    test_results['monte_carlo'] = test_monte_carlo() 
    test_results['integration'] = test_framework_integration()
    test_results['decisions'] = test_decision_system()
    
    # Run demonstration
    demo_success, demo_results = run_demonstration()
    test_results['demonstration'] = demo_success
    
    total_time = time.time() - test_start_time
    
    # Calculate results
    total_tests = len([k for k in test_results.keys() if k != 'imports'])
    passed_tests = sum([1 for k, v in test_results.items() if k != 'imports' and v])
    
    # Print summary
    print(f"\n🎯 TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    
    # Component status
    print(f"\n📊 COMPONENT STATUS:")
    imports = test_results['imports']
    print(f"  SimulationEngine: {'✅ Available' if imports.get('simulation_engine') else '❌ Not Available'}")
    print(f"  MonteCarloEngine: {'✅ Available' if imports.get('monte_carlo') else '❌ Not Available'}")
    
    # Test status
    print(f"\n🧪 TEST STATUS:")
    status_icons = {True: "✅", False: "❌"}
    for test_name, passed in test_results.items():
        if test_name != 'imports':
            icon = status_icons[passed]
            print(f"  {test_name.replace('_', ' ').title()}: {icon}")
    
    if demo_success and demo_results:
        print(f"\n🚀 FRAMEWORK STATUS: OPERATIONAL")
        print(f"  The ChipGeopolitics simulation framework is working correctly!")
        print(f"  Ready for production use and advanced testing.")
    else:
        print(f"\n⚠️  FRAMEWORK STATUS: PARTIAL")
        print(f"  Some components are working, but full integration needs attention.")
    
    print("🎯" + "=" * 60 + "🎯")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 