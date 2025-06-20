"""
Comprehensive Test Runner for ChipGeopolitics Simulation Framework

Runs all test suites including unit tests, integration tests,
performance benchmarks, and validation tests.
"""

import sys
import time
import unittest
import json
import os
from pathlib import Path
from datetime import datetime
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Import test modules
from tests.unit.test_core_components import *
from tests.integration.test_full_simulation import *
from tests.performance.test_benchmarks import *

class TestSuiteRunner:
    """Comprehensive test suite runner with reporting."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_unit_tests(self):
        """Run all unit tests."""
        print("=" * 80)
        print("RUNNING UNIT TESTS")
        print("=" * 80)
        
        # Create unit test suite
        unit_suite = unittest.TestSuite()
        
        # Add unit test cases
        unit_suite.addTest(unittest.makeSuite(TestSimulationEngine))
        unit_suite.addTest(unittest.makeSuite(TestMonteCarloEngine))
        unit_suite.addTest(unittest.makeSuite(TestBaseAgent))
        unit_suite.addTest(unittest.makeSuite(TestAgentState))
        unit_suite.addTest(unittest.makeSuite(TestDecision))
        
        # Run unit tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        unit_result = runner.run(unit_suite)
        
        self.results['unit_tests'] = {
            'tests_run': unit_result.testsRun,
            'failures': len(unit_result.failures),
            'errors': len(unit_result.errors),
            'success_rate': ((unit_result.testsRun - len(unit_result.failures) - len(unit_result.errors)) / unit_result.testsRun * 100) if unit_result.testsRun > 0 else 0,
            'failure_details': [str(test) for test, _ in unit_result.failures],
            'error_details': [str(test) for test, _ in unit_result.errors]
        }
        
        return unit_result
    
    def run_integration_tests(self):
        """Run all integration tests."""
        print("\n" + "=" * 80)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 80)
        
        # Create integration test suite
        integration_suite = unittest.TestSuite()
        
        # Add integration test cases
        integration_suite.addTest(unittest.makeSuite(TestFullSimulationWorkflow))
        integration_suite.addTest(unittest.makeSuite(TestComponentIntegration))
        
        # Run integration tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        integration_result = runner.run(integration_suite)
        
        self.results['integration_tests'] = {
            'tests_run': integration_result.testsRun,
            'failures': len(integration_result.failures),
            'errors': len(integration_result.errors),
            'success_rate': ((integration_result.testsRun - len(integration_result.failures) - len(integration_result.errors)) / integration_result.testsRun * 100) if integration_result.testsRun > 0 else 0,
            'failure_details': [str(test) for test, _ in integration_result.failures],
            'error_details': [str(test) for test, _ in integration_result.errors]
        }
        
        return integration_result
    
    def run_performance_tests(self):
        """Run all performance and validation tests."""
        print("\n" + "=" * 80)
        print("RUNNING PERFORMANCE & VALIDATION TESTS")
        print("=" * 80)
        
        # Create performance test suite
        performance_suite = unittest.TestSuite()
        
        # Add performance test cases
        performance_suite.addTest(unittest.makeSuite(TestSimulationPerformance))
        performance_suite.addTest(unittest.makeSuite(TestDataValidation))
        
        # Run performance tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        performance_result = runner.run(performance_suite)
        
        self.results['performance_tests'] = {
            'tests_run': performance_result.testsRun,
            'failures': len(performance_result.failures),
            'errors': len(performance_result.errors),
            'success_rate': ((performance_result.testsRun - len(performance_result.failures) - len(performance_result.errors)) / performance_result.testsRun * 100) if performance_result.testsRun > 0 else 0,
            'failure_details': [str(test) for test, _ in performance_result.failures],
            'error_details': [str(test) for test, _ in performance_result.errors]
        }
        
        return performance_result
    
    def run_all_tests(self):
        """Run all test suites and generate comprehensive report."""
        print("🚀 CHIPGEOPOLITICS SIMULATION FRAMEWORK - COMPREHENSIVE TEST SUITE")
        print("🔬 Testing all components, integrations, performance, and validation")
        print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        # Run all test categories
        unit_result = self.run_unit_tests()
        integration_result = self.run_integration_tests()
        performance_result = self.run_performance_tests()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Return overall success
        total_failures = (len(unit_result.failures) + len(integration_result.failures) + 
                         len(performance_result.failures))
        total_errors = (len(unit_result.errors) + len(integration_result.errors) + 
                       len(performance_result.errors))
        
        return total_failures == 0 and total_errors == 0
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report."""
        total_execution_time = self.end_time - self.start_time
        
        # Calculate totals
        total_tests = sum([result['tests_run'] for result in self.results.values()])
        total_failures = sum([result['failures'] for result in self.results.values()])
        total_errors = sum([result['errors'] for result in self.results.values()])
        total_success = total_tests - total_failures - total_errors
        overall_success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        # Print summary report
        print("\n" + "🎯" + "=" * 78 + "🎯")
        print("🎯" + " " * 20 + "COMPREHENSIVE TEST REPORT" + " " * 31 + "🎯")
        print("🎯" + "=" * 78 + "🎯")
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   • Total Tests Run: {total_tests}")
        print(f"   • Successful: {total_success}")
        print(f"   • Failed: {total_failures}")
        print(f"   • Errors: {total_errors}")
        print(f"   • Success Rate: {overall_success_rate:.1f}%")
        print(f"   • Total Execution Time: {total_execution_time:.2f} seconds")
        
        print(f"\n📋 BREAKDOWN BY TEST CATEGORY:")
        
        # Category icons
        icons = {
            'unit_tests': '🔧',
            'integration_tests': '🔗',
            'performance_tests': '⚡'
        }
        
        category_names = {
            'unit_tests': 'Unit Tests',
            'integration_tests': 'Integration Tests',
            'performance_tests': 'Performance & Validation'
        }
        
        for category, result in self.results.items():
            icon = icons.get(category, '📝')
            name = category_names.get(category, category.replace('_', ' ').title())
            
            print(f"\n   {icon} {name}:")
            print(f"      Tests: {result['tests_run']}")
            print(f"      Success Rate: {result['success_rate']:.1f}%")
            
            if result['failures'] > 0:
                print(f"      ❌ Failures: {result['failures']}")
                for failure in result['failure_details']:
                    print(f"         - {failure}")
            
            if result['errors'] > 0:
                print(f"      ⚠️  Errors: {result['errors']}")
                for error in result['error_details']:
                    print(f"         - {error}")
        
        # Status summary
        if total_failures == 0 and total_errors == 0:
            print(f"\n🎉 ALL TESTS PASSED! Framework is ready for production use.")
            status = "✅ PASSED"
        elif total_failures == 0:
            print(f"\n⚠️  Some errors occurred but no test failures. Review errors above.")
            status = "⚠️ PASSED WITH WARNINGS"
        else:
            print(f"\n❌ Some tests failed. Please review and fix issues before deployment.")
            status = "❌ FAILED"
        
        print(f"\n🏁 FINAL STATUS: {status}")
        print("🎯" + "=" * 78 + "🎯\n")
        
        # Save detailed results to file
        self.save_test_results(status, total_execution_time)
    
    def save_test_results(self, status, execution_time):
        """Save detailed test results to JSON file."""
        
        # Ensure outputs directory exists
        outputs_dir = Path("tests/outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        detailed_results = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'status': status,
            'summary': {
                'total_tests': sum([result['tests_run'] for result in self.results.values()]),
                'total_failures': sum([result['failures'] for result in self.results.values()]),
                'total_errors': sum([result['errors'] for result in self.results.values()]),
                'overall_success_rate': sum([result['tests_run'] for result in self.results.values()]) - sum([result['failures'] for result in self.results.values()]) - sum([result['errors'] for result in self.results.values()])
            },
            'categories': self.results,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Save to timestamped file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = outputs_dir / f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        # Also save as latest
        latest_file = outputs_dir / "latest_test_results.json"
        with open(latest_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

def main():
    """Main test runner function."""
    # Create and run test suite
    runner = TestSuiteRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 