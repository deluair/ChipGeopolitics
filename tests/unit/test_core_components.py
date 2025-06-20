"""
Unit Tests for Core Components

Basic unit testing for the available components in the
ChipGeopolitics simulation framework.
"""

import unittest
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Test what's actually available
try:
    from core.simulation_engine import SimulationEngine, Decision, AgentState, BaseAgent
    SIMULATION_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SimulationEngine: {e}")
    SIMULATION_ENGINE_AVAILABLE = False

try:
    from core.monte_carlo import MonteCarloEngine
    MONTE_CARLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MonteCarloEngine: {e}")
    MONTE_CARLO_AVAILABLE = False


class TestSimulationEngine(unittest.TestCase):
    """Test cases for SimulationEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("SimulationEngine not available")
        self.engine = SimulationEngine()
    
    def test_engine_initialization(self):
        """Test simulation engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.current_step, 0)
        self.assertEqual(len(self.engine.agents), 0)
        self.assertFalse(self.engine.running)
    
    def test_add_agent(self):
        """Test adding agents to simulation."""
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        
        self.engine.add_agent(mock_agent)
        self.assertEqual(len(self.engine.agents), 1)
        self.assertIn("test_agent", self.engine.agents)
    
    def test_remove_agent(self):
        """Test removing agents from simulation."""
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        
        self.engine.add_agent(mock_agent)
        self.engine.remove_agent("test_agent")
        self.assertEqual(len(self.engine.agents), 0)
    
    def test_step_execution(self):
        """Test single simulation step execution."""
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.step.return_value = Decision("test_action", {}, 0.5)
        
        self.engine.add_agent(mock_agent)
        self.engine.step()
        
        self.assertEqual(self.engine.current_step, 1)
        mock_agent.step.assert_called_once()
    
    def test_simulation_run(self):
        """Test basic simulation run."""
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.step.return_value = Decision("test_action", {}, 0.5)
        
        self.engine.add_agent(mock_agent)
        
        # Run for 5 steps
        results = self.engine.run(steps=5)
        
        self.assertEqual(self.engine.current_step, 5)
        self.assertIsInstance(results, dict)
        self.assertIn("final_step", results)


class TestMonteCarloEngine(unittest.TestCase):
    """Test cases for MonteCarloEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MONTE_CARLO_AVAILABLE:
            self.skipTest("MonteCarloEngine not available")
        self.mc_engine = MonteCarloEngine()
    
    def test_engine_initialization(self):
        """Test Monte Carlo engine initialization."""
        self.assertIsNotNone(self.mc_engine)
        self.assertEqual(self.mc_engine.num_runs, 1000)
        self.assertIsNone(self.mc_engine.simulation_engine)
    
    def test_set_simulation_engine(self):
        """Test setting simulation engine."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("SimulationEngine not available")
        
        sim_engine = SimulationEngine()
        self.mc_engine.set_simulation_engine(sim_engine)
        self.assertEqual(self.mc_engine.simulation_engine, sim_engine)
    
    def test_parameter_sampling(self):
        """Test parameter sampling for Monte Carlo runs."""
        parameters = {
            "market_growth": ("normal", 0.08, 0.02),
            "geopolitical_tension": ("uniform", 0.3, 0.7)
        }
        
        samples = self.mc_engine.sample_parameters(parameters, num_samples=10)
        
        self.assertEqual(len(samples), 10)
        self.assertIn("market_growth", samples[0])
        self.assertIn("geopolitical_tension", samples[0])


class TestBaseAgent(unittest.TestCase):
    """Test cases for BaseAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("BaseAgent not available")
        self.agent = BaseAgent(
            agent_id="test_agent",
            agent_type="test",
            initial_state={"resource": 100, "capability": 0.8}
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_id, "test_agent")
        self.assertEqual(self.agent.agent_type, "test")
        self.assertEqual(self.agent.state.data["resource"], 100)
        self.assertEqual(self.agent.state.data["capability"], 0.8)
    
    def test_state_updates(self):
        """Test agent state updates."""
        self.agent.update_state("resource", 150)
        self.assertEqual(self.agent.state.data["resource"], 150)
        
        self.agent.update_state("new_field", "test_value")
        self.assertEqual(self.agent.state.data["new_field"], "test_value")
    
    def test_decision_making(self):
        """Test agent decision making process."""
        decision = self.agent.step()
        
        self.assertIsInstance(decision, Decision)
        self.assertEqual(decision.action, "default_action")
        self.assertEqual(decision.utility, 0.5)
    
    def test_state_validation(self):
        """Test agent state validation."""
        # Test valid state
        valid_state = {"resource": 50, "capability": 0.9}
        self.assertTrue(self.agent.validate_state(valid_state))
        
        # Test invalid state (negative resource)
        invalid_state = {"resource": -10, "capability": 0.9}
        self.assertFalse(self.agent.validate_state(invalid_state))


class TestAgentState(unittest.TestCase):
    """Test cases for AgentState."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("AgentState not available")
        self.state = AgentState({
            "resource": 100,
            "capability": 0.8,
            "market_position": 0.6
        })
    
    def test_state_initialization(self):
        """Test state initialization."""
        self.assertEqual(self.state.data["resource"], 100)
        self.assertEqual(self.state.data["capability"], 0.8)
        self.assertIsInstance(self.state.timestamp, datetime)
    
    def test_state_updates(self):
        """Test state update operations."""
        original_timestamp = self.state.timestamp
        
        self.state.update("resource", 150)
        
        self.assertEqual(self.state.data["resource"], 150)
        self.assertGreater(self.state.timestamp, original_timestamp)
    
    def test_state_history(self):
        """Test state change history tracking."""
        self.state.update("resource", 120)
        self.state.update("capability", 0.9)
        
        history = self.state.get_history()
        
        self.assertGreaterEqual(len(history), 2)
    
    def test_state_serialization(self):
        """Test state serialization and deserialization."""
        state_dict = self.state.to_dict()
        
        self.assertIn("data", state_dict)
        self.assertIn("timestamp", state_dict)
        self.assertEqual(state_dict["data"]["resource"], 100)
        
        # Test deserialization
        new_state = AgentState.from_dict(state_dict)
        self.assertEqual(new_state.data["resource"], 100)
        self.assertEqual(new_state.data["capability"], 0.8)


class TestDecision(unittest.TestCase):
    """Test cases for Decision class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("Decision not available")
    
    def test_decision_creation(self):
        """Test decision object creation."""
        decision = Decision("invest", {"amount": 1000, "target": "AI"}, 0.85)
        
        self.assertEqual(decision.action, "invest")
        self.assertEqual(decision.parameters["amount"], 1000)
        self.assertEqual(decision.parameters["target"], "AI")
        self.assertEqual(decision.utility, 0.85)
        self.assertIsInstance(decision.timestamp, datetime)
    
    def test_decision_validation(self):
        """Test decision validation."""
        # Valid decision
        valid_decision = Decision("valid_action", {"param": 1}, 0.5)
        self.assertTrue(valid_decision.is_valid())
        
        # Invalid decision (utility out of range)
        invalid_decision = Decision("invalid_action", {"param": 1}, 1.5)
        self.assertFalse(invalid_decision.is_valid())
    
    def test_decision_comparison(self):
        """Test decision comparison by utility."""
        decision1 = Decision("action1", {}, 0.7)
        decision2 = Decision("action2", {}, 0.9)
        
        self.assertLess(decision1, decision2)
        self.assertGreater(decision2, decision1)
    
    def test_decision_serialization(self):
        """Test decision serialization."""
        decision = Decision("test_action", {"key": "value"}, 0.6)
        decision_dict = decision.to_dict()
        
        self.assertEqual(decision_dict["action"], "test_action")
        self.assertEqual(decision_dict["parameters"]["key"], "value")
        self.assertEqual(decision_dict["utility"], 0.6)


class TestFrameworkIntegration(unittest.TestCase):
    """Test basic framework integration."""
    
    def test_simulation_with_mock_agents(self):
        """Test running simulation with mock agents."""
        if not SIMULATION_ENGINE_AVAILABLE:
            self.skipTest("SimulationEngine not available")
        
        engine = SimulationEngine()
        
        # Create mock agents
        for i in range(3):
            mock_agent = Mock()
            mock_agent.agent_id = f"agent_{i}"
            mock_agent.step.return_value = Decision(f"action_{i}", {}, 0.5 + i * 0.1)
            engine.add_agent(mock_agent)
        
        # Run simulation
        results = engine.run(steps=5)
        
        # Verify results
        self.assertIn("final_step", results)
        self.assertEqual(results["final_step"], 5)
        self.assertEqual(results["agent_count"], 3)
        self.assertIn("agent_states", results)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases based on availability
    if SIMULATION_ENGINE_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestSimulationEngine))
        suite.addTest(unittest.makeSuite(TestBaseAgent))
        suite.addTest(unittest.makeSuite(TestAgentState))
        suite.addTest(unittest.makeSuite(TestDecision))
        suite.addTest(unittest.makeSuite(TestFrameworkIntegration))
    
    if MONTE_CARLO_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestMonteCarloEngine))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print("Failures:")
        for test, error in result.failures:
            print(f"  - {test}: {error}")
    
    if result.errors:
        print("Errors:")
        for test, error in result.errors:
            print(f"  - {test}: {error}") 