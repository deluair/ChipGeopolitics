"""
Performance Benchmarks and Validation Tests

Comprehensive performance testing including scalability benchmarks,
memory usage analysis, and validation against known datasets.
"""

import unittest
import sys
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import gc
import threading
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
sys.path.append('src')

from core.simulation_engine import SimulationEngine
from core.monte_carlo import MonteCarloEngine
from agents.chip_manufacturer import ChipManufacturerAgent
from agents.hyperscaler import HyperscalerAgent
from market.demand_models import DemandForecastModel
from supply_chain.critical_path import CriticalPathAnalyzer
from analytics.performance_tracker import PerformanceTracker

class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return {
            "result": result,
            "execution_time": end_time - start_time,
            "timestamp": datetime.now()
        }
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage during function execution."""
        # Force garbage collection before measurement
        gc.collect()
        
        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get final memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            "result": result,
            "execution_time": end_time - start_time,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_delta_mb": final_memory - initial_memory,
            "peak_memory_mb": max(initial_memory, final_memory)
        }
    
    def run_scalability_test(self, func, scale_params, iterations=3):
        """Run scalability test across different parameter scales."""
        results = {}
        
        for scale in scale_params:
            scale_results = []
            
            for i in range(iterations):
                gc.collect()  # Clean up before each run
                
                measurement = self.measure_memory_usage(func, scale)
                scale_results.append(measurement)
            
            # Calculate statistics
            exec_times = [r["execution_time"] for r in scale_results]
            memory_deltas = [r["memory_delta_mb"] for r in scale_results]
            
            results[scale] = {
                "avg_execution_time": np.mean(exec_times),
                "std_execution_time": np.std(exec_times),
                "min_execution_time": np.min(exec_times),
                "max_execution_time": np.max(exec_times),
                "avg_memory_delta": np.mean(memory_deltas),
                "max_memory_usage": max([r["peak_memory_mb"] for r in scale_results])
            }
        
        return results

class TestSimulationPerformance(unittest.TestCase):
    """Performance tests for simulation components."""
    
    def setUp(self):
        """Set up performance testing environment."""
        self.benchmark = PerformanceBenchmark()
        self.performance_tracker = PerformanceTracker()
        
    def test_simulation_engine_scalability(self):
        """Test simulation engine performance with varying numbers of agents."""
        print("\n=== Testing Simulation Engine Scalability ===")
        
        def run_simulation_with_agents(num_agents):
            """Run simulation with specified number of agents."""
            engine = SimulationEngine()
            
            # Create agents
            for i in range(num_agents):
                agent = ChipManufacturerAgent(
                    agent_id=f"manufacturer_{i}",
                    company_name=f"Company_{i}",
                    initial_state={
                        "manufacturing_capacity": 500 + i * 100,
                        "technology_level": 0.8 + (i % 3) * 0.05,
                        "market_share": 0.1 + (i % 5) * 0.02
                    }
                )
                engine.add_agent(agent)
            
            # Run simulation
            return engine.run(steps=10)
        
        # Test with different numbers of agents
        agent_counts = [1, 5, 10, 20, 50]
        
        scalability_results = self.benchmark.run_scalability_test(
            run_simulation_with_agents,
            agent_counts,
            iterations=3
        )
        
        # Analyze results
        print(f"{'Agents':<8} {'Avg Time (s)':<12} {'Memory (MB)':<12} {'Efficiency':<12}")
        print("-" * 50)
        
        baseline_time = scalability_results[1]["avg_execution_time"]
        
        for agent_count in agent_counts:
            result = scalability_results[agent_count]
            efficiency = baseline_time * agent_count / result["avg_execution_time"]
            
            print(f"{agent_count:<8} "
                  f"{result['avg_execution_time']:<12.3f} "
                  f"{result['max_memory_usage']:<12.1f} "
                  f"{efficiency:<12.2f}")
        
        # Performance assertions
        # Time should scale sub-linearly with agent count
        time_50_agents = scalability_results[50]["avg_execution_time"]
        time_1_agent = scalability_results[1]["avg_execution_time"]
        
        # Should not be more than 100x slower for 50x agents
        self.assertLess(time_50_agents, time_1_agent * 100)
        
        # Memory usage should be reasonable
        memory_50_agents = scalability_results[50]["max_memory_usage"]
        self.assertLess(memory_50_agents, 2000)  # Should use less than 2GB
    
    def test_monte_carlo_performance(self):
        """Test Monte Carlo engine performance."""
        print("\n=== Testing Monte Carlo Performance ===")
        
        def run_monte_carlo_simulation(num_runs):
            """Run Monte Carlo simulation with specified number of runs."""
            mc_engine = MonteCarloEngine()
            sim_engine = SimulationEngine()
            
            # Add a test agent
            agent = ChipManufacturerAgent(
                agent_id="test_agent",
                company_name="TestCorp",
                initial_state={
                    "manufacturing_capacity": 1000,
                    "technology_level": 0.9,
                    "market_share": 0.3
                }
            )
            sim_engine.add_agent(agent)
            mc_engine.set_simulation_engine(sim_engine)
            
            # Define parameters
            parameters = {
                "market_growth": ("normal", 0.08, 0.02),
                "geopolitical_tension": ("uniform", 0.2, 0.8)
            }
            
            # Run Monte Carlo
            return mc_engine.run_monte_carlo(
                parameters=parameters,
                num_runs=num_runs,
                max_steps_per_run=5,
                output_metrics=["market_size", "risk_level"]
            )
        
        # Test with different numbers of runs
        run_counts = [10, 50, 100, 250]
        
        mc_results = self.benchmark.run_scalability_test(
            run_monte_carlo_simulation,
            run_counts,
            iterations=2
        )
        
        # Analyze results
        print(f"{'Runs':<8} {'Avg Time (s)':<12} {'Time/Run (ms)':<15} {'Memory (MB)':<12}")
        print("-" * 55)
        
        for run_count in run_counts:
            result = mc_results[run_count]
            time_per_run = (result["avg_execution_time"] / run_count) * 1000
            
            print(f"{run_count:<8} "
                  f"{result['avg_execution_time']:<12.3f} "
                  f"{time_per_run:<15.2f} "
                  f"{result['max_memory_usage']:<12.1f}")
        
        # Performance assertions
        # Time per run should be relatively stable
        time_per_run_10 = mc_results[10]["avg_execution_time"] / 10
        time_per_run_250 = mc_results[250]["avg_execution_time"] / 250
        
        # Should not be more than 3x different
        ratio = max(time_per_run_10, time_per_run_250) / min(time_per_run_10, time_per_run_250)
        self.assertLess(ratio, 3.0)
    
    def test_market_model_performance(self):
        """Test market model performance with large datasets."""
        print("\n=== Testing Market Model Performance ===")
        
        def run_demand_forecast(time_horizon):
            """Run demand forecast with specified time horizon."""
            demand_model = DemandForecastModel()
            
            return demand_model.generate_demand_forecast(
                time_horizon=time_horizon,
                scenario_params={
                    "ai_growth_factor": 1.5,
                    "economic_growth": 0.04,
                    "technology_adoption_rate": 0.25
                }
            )
        
        # Test with different time horizons
        time_horizons = [12, 24, 60, 120]
        
        market_results = self.benchmark.run_scalability_test(
            run_demand_forecast,
            time_horizons,
            iterations=3
        )
        
        # Analyze results
        print(f"{'Horizon':<8} {'Avg Time (s)':<12} {'Memory (MB)':<12}")
        print("-" * 35)
        
        for horizon in time_horizons:
            result = market_results[horizon]
            
            print(f"{horizon:<8} "
                  f"{result['avg_execution_time']:<12.3f} "
                  f"{result['max_memory_usage']:<12.1f}")
        
        # Performance assertions
        # Should complete forecast within reasonable time
        for horizon in time_horizons:
            self.assertLess(market_results[horizon]["avg_execution_time"], 10.0)
    
    def test_supply_chain_analysis_performance(self):
        """Test supply chain analysis performance."""
        print("\n=== Testing Supply Chain Analysis Performance ===")
        
        def run_supply_chain_analysis(complexity_level):
            """Run supply chain analysis with varying complexity."""
            analyzer = CriticalPathAnalyzer()
            
            # Configure complexity (simplified for testing)
            if complexity_level == "simple":
                analyzer.max_paths_to_analyze = 10
            elif complexity_level == "medium":
                analyzer.max_paths_to_analyze = 50
            elif complexity_level == "complex":
                analyzer.max_paths_to_analyze = 100
            
            return analyzer.analyze_critical_paths()
        
        complexity_levels = ["simple", "medium", "complex"]
        
        for level in complexity_levels:
            measurement = self.benchmark.measure_memory_usage(
                run_supply_chain_analysis, level
            )
            
            print(f"Complexity: {level}")
            print(f"  Execution time: {measurement['execution_time']:.3f}s")
            print(f"  Memory usage: {measurement['memory_delta_mb']:.1f}MB")
            
            # Performance assertions
            self.assertLess(measurement["execution_time"], 30.0)  # Should complete within 30s
            self.assertLess(measurement["memory_delta_mb"], 500)  # Should use less than 500MB
    
    def test_concurrent_simulation_performance(self):
        """Test performance of concurrent simulations."""
        print("\n=== Testing Concurrent Simulation Performance ===")
        
        def single_simulation():
            """Run a single simulation."""
            engine = SimulationEngine()
            agent = ChipManufacturerAgent(
                agent_id="concurrent_test",
                company_name="ConcurrentCorp",
                initial_state={
                    "manufacturing_capacity": 800,
                    "technology_level": 0.85,
                    "market_share": 0.25
                }
            )
            engine.add_agent(agent)
            return engine.run(steps=5)
        
        # Test sequential execution
        start_time = time.time()
        sequential_results = []
        for i in range(4):
            sequential_results.append(single_simulation())
        sequential_time = time.time() - start_time
        
        # Test concurrent execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_futures = [executor.submit(single_simulation) for _ in range(4)]
            concurrent_results = [f.result() for f in concurrent_futures]
        concurrent_time = time.time() - start_time
        
        print(f"Sequential execution: {sequential_time:.3f}s")
        print(f"Concurrent execution: {concurrent_time:.3f}s")
        print(f"Speedup: {sequential_time / concurrent_time:.2f}x")
        
        # Verify results are valid
        self.assertEqual(len(sequential_results), 4)
        self.assertEqual(len(concurrent_results), 4)
        
        # Concurrent should be faster (or at least not much slower)
        self.assertLess(concurrent_time, sequential_time * 1.5)
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during long simulations."""
        print("\n=== Testing Memory Leak Detection ===")
        
        initial_memory = self.benchmark.process.memory_info().rss / 1024 / 1024
        memory_samples = [initial_memory]
        
        # Run multiple simulation cycles
        for cycle in range(5):
            engine = SimulationEngine()
            
            # Add agents
            for i in range(10):
                agent = ChipManufacturerAgent(
                    agent_id=f"leak_test_{cycle}_{i}",
                    company_name=f"LeakTest_{i}",
                    initial_state={
                        "manufacturing_capacity": 500,
                        "technology_level": 0.8,
                        "market_share": 0.1
                    }
                )
                engine.add_agent(agent)
            
            # Run simulation
            engine.run(steps=20)
            
            # Force cleanup
            del engine
            gc.collect()
            
            # Sample memory
            current_memory = self.benchmark.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            print(f"Cycle {cycle + 1}: {current_memory:.1f}MB")
        
        # Analyze memory trend
        memory_increase = memory_samples[-1] - memory_samples[0]
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        
        print(f"Total memory increase: {memory_increase:.1f}MB")
        print(f"Memory trend: {memory_trend:.3f}MB per cycle")
        
        # Memory increase should be reasonable (less than 200MB)
        self.assertLess(memory_increase, 200)
        
        # Memory trend should be minimal (less than 10MB per cycle)
        self.assertLess(abs(memory_trend), 10)

class TestDataValidation(unittest.TestCase):
    """Validation tests against known datasets and expected behaviors."""
    
    def setUp(self):
        """Set up validation testing environment."""
        self.tolerance = 0.1  # 10% tolerance for validation
    
    def test_market_forecast_validation(self):
        """Validate market forecasts against known industry data."""
        print("\n=== Validating Market Forecasts ===")
        
        demand_model = DemandForecastModel()
        
        # Generate forecast for known scenario
        forecast = demand_model.generate_demand_forecast(
            time_horizon=12,
            scenario_params={
                "ai_growth_factor": 1.0,  # No AI boom
                "economic_growth": 0.03,   # Standard growth
                "technology_adoption_rate": 0.2
            }
        )
        
        # Validate total market size is reasonable
        total_demand = forecast["total_demand"]
        
        # Global semiconductor market is ~$500-600B
        self.assertGreater(total_demand, 400)
        self.assertLess(total_demand, 800)
        
        # Validate segment breakdown sums to total
        segment_sum = sum(forecast["segment_breakdown"].values())
        self.assertAlmostEqual(segment_sum, total_demand, delta=total_demand * 0.01)
        
        # Validate time series has expected length
        self.assertEqual(len(forecast["time_series"]), 12)
        
        print(f"Total market demand: ${total_demand:.1f}B")
        print("Segment breakdown:")
        for segment, value in forecast["segment_breakdown"].items():
            print(f"  {segment}: ${value:.1f}B ({value/total_demand*100:.1f}%)")
    
    def test_supply_chain_realism(self):
        """Validate supply chain analysis against realistic constraints."""
        print("\n=== Validating Supply Chain Realism ===")
        
        analyzer = CriticalPathAnalyzer()
        analysis = analyzer.analyze_critical_paths()
        
        # Validate critical paths exist
        critical_paths = analysis["critical_paths"]
        self.assertGreater(len(critical_paths), 0)
        
        # Validate path timing is realistic
        for path in critical_paths:
            total_time = path["total_time"]
            
            # Semiconductor manufacturing cycles typically 2-6 months
            self.assertGreater(total_time, 60)    # At least 2 months
            self.assertLess(total_time, 365)      # Less than 1 year
            
            # Risk scores should be between 0 and 1
            risk_score = path["risk_score"]
            self.assertGreaterEqual(risk_score, 0.0)
            self.assertLessEqual(risk_score, 1.0)
        
        # Validate bottlenecks are identified
        bottlenecks = analysis["bottlenecks"]
        self.assertGreater(len(bottlenecks), 0)
        
        print(f"Critical paths identified: {len(critical_paths)}")
        print(f"Bottlenecks found: {len(bottlenecks)}")
        print(f"Average path time: {np.mean([p['total_time'] for p in critical_paths]):.1f} days")
    
    def test_agent_behavior_validation(self):
        """Validate that agent behaviors are realistic."""
        print("\n=== Validating Agent Behaviors ===")
        
        # Create manufacturer agent
        manufacturer = ChipManufacturerAgent(
            agent_id="validation_test",
            company_name="ValidationCorp",
            initial_state={
                "manufacturing_capacity": 1000,
                "technology_level": 0.90,
                "market_share": 0.30
            }
        )
        
        # Test decision making over multiple steps
        decisions = []
        for step in range(10):
            decision = manufacturer.step()
            decisions.append(decision)
        
        # Validate decision consistency
        actions = [d.action for d in decisions]
        utilities = [d.utility for d in decisions]
        
        # Should have valid actions
        valid_actions = manufacturer.get_available_actions()
        for action in actions:
            self.assertIn(action, valid_actions)
        
        # Utilities should be in valid range
        for utility in utilities:
            self.assertGreaterEqual(utility, 0.0)
            self.assertLessEqual(utility, 1.0)
        
        # Should show some variation in decision making
        unique_actions = set(actions)
        self.assertGreater(len(unique_actions), 1)
        
        print(f"Unique actions taken: {len(unique_actions)}")
        print(f"Average utility: {np.mean(utilities):.3f}")
        print(f"Utility variance: {np.var(utilities):.3f}")
    
    def test_geopolitical_impact_validation(self):
        """Validate geopolitical impact calculations."""
        print("\n=== Validating Geopolitical Impacts ===")
        
        from geopolitical.export_controls import ExportControlSimulator
        
        export_sim = ExportControlSimulator()
        
        # Simulate realistic export control scenario
        impact = export_sim.simulate_export_controls(
            target_countries=["china"],
            restricted_technologies=["advanced_semiconductors"],
            restriction_level=0.7,
            duration_months=24
        )
        
        # Validate impact structure
        self.assertIn("market_impact", impact)
        self.assertIn("supply_chain_disruption", impact)
        self.assertIn("affected_companies", impact)
        
        # Validate impact magnitudes are realistic
        market_impact = impact["market_impact"]
        revenue_impact = market_impact["revenue_impact"]
        
        # Revenue impact should be significant but not total collapse
        self.assertGreater(abs(revenue_impact), 0.05)  # At least 5% impact
        self.assertLess(abs(revenue_impact), 0.5)      # Less than 50% impact
        
        # Validate affected companies list
        affected_companies = impact["affected_companies"]
        self.assertGreater(len(affected_companies), 0)
        
        print(f"Revenue impact: {revenue_impact:.1%}")
        print(f"Companies affected: {len(affected_companies)}")
        print(f"Supply chain disruption: {impact['supply_chain_disruption']['severity']:.2f}")

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add performance test cases
    suite.addTest(unittest.makeSuite(TestSimulationPerformance))
    suite.addTest(unittest.makeSuite(TestDataValidation))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    
    print("=" * 80)
    print("CHIPGEOPOLITICS PERFORMANCE BENCHMARKS & VALIDATION")
    print("=" * 80)
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"PERFORMANCE & VALIDATION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successful: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print(f"{'='*80}")
    
    # System info
    print(f"\nSystem Information:")
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"Memory Total: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
    print(f"Python Version: {sys.version}")
    print(f"Test Platform: {sys.platform}") 