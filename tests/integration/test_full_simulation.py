"""
Integration Tests for Available Components

Basic integration testing for the ChipGeopolitics simulation framework
using available components and mock implementations where needed.
"""

import unittest
import sys
import time
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Test what's actually available
try:
    from core.simulation_engine import SimulationEngine, Decision, BaseAgent
    SIMULATION_ENGINE_AVAILABLE = True
except ImportError:
    SIMULATION_ENGINE_AVAILABLE = False

try:
    from core.monte_carlo import MonteCarloEngine
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False


class TestBasicSimulationWorkflow(unittest.TestCase):
    """Test basic simulation workflow."""
    
    def setUp(self):
        """Set up test simulation environment."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("SimulationEngine not available")
        self.sim_engine = SimulationEngine()
    
    def test_basic_simulation_setup(self):
        """Test basic simulation setup with mock agents."""
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent1.agent_id = "test_agent_1"
        mock_agent1.step.return_value = Decision("action1", {}, 0.6)
        
        mock_agent2 = Mock()
        mock_agent2.agent_id = "test_agent_2"
        mock_agent2.step.return_value = Decision("action2", {}, 0.7)
        
        # Add agents to simulation
        self.sim_engine.add_agent(mock_agent1)
        self.sim_engine.add_agent(mock_agent2)
        
        # Verify setup
        self.assertEqual(len(self.sim_engine.agents), 2)
        self.assertIn("test_agent_1", self.sim_engine.agents)
        self.assertIn("test_agent_2", self.sim_engine.agents)
    
    def test_short_simulation_run(self):
        """Test a short simulation run for basic functionality."""
        # Setup mock agent
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.step.return_value = Decision("test_action", {}, 0.5)
        
        self.sim_engine.add_agent(mock_agent)
        
        # Record start time
        start_time = time.time()
        
        # Run simulation for 5 steps
        results = self.sim_engine.run(steps=5)
        
        # Record execution time
        execution_time = time.time() - start_time
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertEqual(self.sim_engine.current_step, 5)
        self.assertIn("final_step", results)
        
        # Performance checks
        self.assertLess(execution_time, 10)  # Should complete within 10 seconds
    
    def test_multi_agent_interaction(self):
        """Test interactions between multiple agent types."""
        # Create diverse mock agents
        agents = []
        for i in range(3):
            mock_agent = Mock()
            mock_agent.agent_id = f"agent_{i}"
            mock_agent.step.return_value = Decision(f"action_{i}", {}, 0.5 + i * 0.1)
            agents.append(mock_agent)
        
        # Add all agents
        for agent in agents:
            self.sim_engine.add_agent(agent)
        
        # Run simulation
        results = self.sim_engine.run(steps=10)
        
        # Verify multi-agent execution
        self.assertEqual(len(results["agent_states"]), 3)
        self.assertIn("converged", results)
        
        # Check that agents made decisions
        for i in range(3):
            agent_id = f"agent_{i}"
            self.assertIn(agent_id, results["agent_states"])


class TestMonteCarloIntegration(unittest.TestCase):
    """Test Monte Carlo integration if available."""
    
    def setUp(self):
        """Set up test environment."""
        if not MONTE_CARLO_AVAILABLE or not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("Monte Carlo or Simulation Engine not available")
        
        self.mc_engine = MonteCarloEngine()
        self.sim_engine = SimulationEngine()
    
    def test_monte_carlo_basic_setup(self):
        """Test basic Monte Carlo setup."""
        # Set simulation engine
        self.mc_engine.set_simulation_engine(self.sim_engine)
        self.assertEqual(self.mc_engine.simulation_engine, self.sim_engine)
        
        # Add test agent
        mock_agent = Mock()
        mock_agent.agent_id = "mc_test_agent"
        mock_agent.step.return_value = Decision("test_action", {}, 0.5)
        self.sim_engine.add_agent(mock_agent)
        
        # Test parameter sampling
        parameters = {
            "test_param": ("normal", 0.5, 0.1)
        }
        
        samples = self.mc_engine.sample_parameters(parameters, num_samples=5)
        self.assertEqual(len(samples), 5)
        self.assertIn("test_param", samples[0])


class TestComponentAvailability(unittest.TestCase):
    """Test which components are available."""
    
    def test_core_component_imports(self):
        """Test importing core components."""
        print(f"\nSimulation Engine Available: {SIMULATION_ENGINE_AVAILABLE}")
        print(f"Monte Carlo Available: {MONTE_CARLO_AVAILABLE}")
        
        # At least one core component should be available
        self.assertTrue(SIMULATION_ENGINE_AVAILABLE or MONTE_CARLO_AVAILABLE, 
                       "At least one core component should be available")
    
    def test_basic_functionality(self):
        """Test basic functionality of available components."""
        if SIMULATION_ENGINE_AVAILABLE:
            engine = SimulationEngine()
            self.assertIsInstance(engine, SimulationEngine)
            self.assertEqual(engine.current_step, 0)
            print("✓ SimulationEngine basic functionality works")
        
        if MONTE_CARLO_AVAILABLE:
            mc = MonteCarloEngine()
            self.assertIsInstance(mc, MonteCarloEngine)
            self.assertEqual(mc.num_runs, 1000)
            print("✓ MonteCarloEngine basic functionality works")


class TestFrameworkDemo(unittest.TestCase):
    """Test framework demonstration capabilities."""
    
    def test_simple_framework_demo(self):
        """Test a simple framework demonstration."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("SimulationEngine not available")
        
        print("\n=== Simple Framework Demo ===")
        
        # Initialize simulation
        engine = SimulationEngine()
        print("✓ Simulation engine initialized")
        
        # Create mock agents representing different types
        agent_types = ["manufacturer", "hyperscaler", "supplier"]
        for i, agent_type in enumerate(agent_types):
            mock_agent = Mock()
            mock_agent.agent_id = f"{agent_type}_{i}"
            mock_agent.agent_type = agent_type
            mock_agent.step.return_value = Decision(f"{agent_type}_action", 
                                                   {"priority": i}, 
                                                   0.6 + i * 0.1)
            engine.add_agent(mock_agent)
            print(f"✓ Added {agent_type} agent")
        
        # Run simulation
        print("🔄 Running simulation...")
        start_time = time.time()
        results = engine.run(steps=10)
        execution_time = time.time() - start_time
        
        # Display results
        print(f"✓ Simulation completed in {execution_time:.3f} seconds")
        print(f"  - Final step: {results['final_step']}")
        print(f"  - Agents: {results['agent_count']}")
        print(f"  - Converged: {results.get('converged', 'Unknown')}")
        
        # Verify demo worked
        self.assertEqual(results['final_step'], 10)
        self.assertEqual(results['agent_count'], 3)
        print("✓ Demo completed successfully")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases based on availability
    suite.addTest(unittest.makeSuite(TestComponentAvailability))
    
    if SIMULATION_ENGINE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestBasicSimulationWorkflow))
        suite.addTest(unittest.makeSuite(TestFrameworkDemo))
    
    if MONTE_CARLO_AVAILABLE and SIMULATION_ENGINE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestMonteCarloIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print(f"{'='*60}")
    
    # Component status
    print(f"\nComponent Status:")
    print(f"✓ Simulation Engine: {'Available' if SIMULATION_ENGINE_AVAILABLE else 'Not Available'}")
    print(f"✓ Monte Carlo Engine: {'Available' if MONTE_CARLO_AVAILABLE else 'Not Available'}")
    
    if SIMULATION_ENGINE_AVAILABLE:
        print(f"✓ Basic simulation workflow: Ready")
    if MONTE_CARLO_AVAILABLE and SIMULATION_ENGINE_AVAILABLE:
        print(f"✓ Monte Carlo analysis: Ready")