"""
Monte Carlo probabilistic modeling for ChipGeopolitics simulation.
Handles uncertainty quantification and scenario analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Types of probability distributions supported."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular" 
    BETA = "beta"
    GAMMA = "gamma"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"


@dataclass
class DistributionParams:
    """Parameters for probability distributions."""
    dist_type: DistributionType
    params: Dict[str, float]
    
    def sample(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the distribution."""
        if random_state is not None:
            np.random.seed(random_state)
            
        if self.dist_type == DistributionType.NORMAL:
            return np.random.normal(self.params['mean'], self.params['std'], size)
        elif self.dist_type == DistributionType.UNIFORM:
            return np.random.uniform(self.params['low'], self.params['high'], size)
        elif self.dist_type == DistributionType.TRIANGULAR:
            return np.random.triangular(
                self.params['left'], self.params['mode'], self.params['right'], size
            )
        elif self.dist_type == DistributionType.BETA:
            return np.random.beta(self.params['alpha'], self.params['beta'], size)
        elif self.dist_type == DistributionType.GAMMA:
            return np.random.gamma(self.params['shape'], self.params['scale'], size)
        elif self.dist_type == DistributionType.LOGNORMAL:
            return np.random.lognormal(self.params['mean'], self.params['sigma'], size)
        elif self.dist_type == DistributionType.EXPONENTIAL:
            return np.random.exponential(self.params['scale'], size)
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist_type}")


@dataclass
class ScenarioDefinition:
    """Definition of a scenario for Monte Carlo analysis."""
    name: str
    description: str
    variable_distributions: Dict[str, DistributionParams]
    probability: float = 1.0  # Scenario probability weight
    conditions: Dict[str, Any] = None  # Special conditions for this scenario


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for uncertainty quantification.
    
    Supports:
    - Multiple probability distributions
    - Scenario-based analysis
    - Sensitivity analysis
    - Parallel execution
    - Risk metrics calculation
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Monte Carlo engine.
        
        Args:
            random_seed: Seed for random number generation
        """
        self.random_seed = random_seed
        self.scenarios: List[ScenarioDefinition] = []
        self.results_cache: Dict[str, pd.DataFrame] = {}
        
        # Risk assessment thresholds
        self.var_confidence_levels = [0.95, 0.99, 0.999]  # Value at Risk confidence levels
        self.expected_shortfall_alpha = 0.05  # Expected shortfall alpha
        
        np.random.seed(random_seed)
    
    def add_scenario(self, scenario: ScenarioDefinition) -> None:
        """Add a scenario definition."""
        self.scenarios.append(scenario)
        logger.info(f"Added scenario: {scenario.name}")
    
    def run_simulation(self, 
                      model_function: Callable,
                      n_iterations: int = 1000,
                      scenario_name: Optional[str] = None,
                      parallel: bool = True,
                      max_workers: int = 4) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.
        
        Args:
            model_function: Function that takes sampled parameters and returns results
            n_iterations: Number of Monte Carlo iterations
            scenario_name: Specific scenario to run (if None, runs all scenarios)
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
            
        Returns:
            DataFrame with simulation results
        """
        scenarios_to_run = [s for s in self.scenarios if scenario_name is None or s.name == scenario_name]
        
        if not scenarios_to_run:
            raise ValueError(f"No scenarios found for: {scenario_name}")
        
        all_results = []
        
        for scenario in scenarios_to_run:
            logger.info(f"Running scenario: {scenario.name} with {n_iterations} iterations")
            
            if parallel and max_workers > 1:
                results = self._run_parallel_simulation(
                    model_function, scenario, n_iterations, max_workers
                )
            else:
                results = self._run_sequential_simulation(
                    model_function, scenario, n_iterations
                )
            
            # Add scenario metadata
            results['scenario'] = scenario.name
            results['scenario_probability'] = scenario.probability
            all_results.append(results)
        
        combined_results = pd.concat(all_results, ignore_index=True)
        self.results_cache[scenario_name or 'all_scenarios'] = combined_results
        
        return combined_results
    
    def _run_sequential_simulation(self, 
                                  model_function: Callable,
                                  scenario: ScenarioDefinition,
                                  n_iterations: int) -> pd.DataFrame:
        """Run simulation sequentially."""
        results = []
        
        for i in range(n_iterations):
            # Sample parameters for this iteration
            sampled_params = self._sample_scenario_parameters(scenario, iteration=i)
            
            # Run model with sampled parameters
            try:
                model_result = model_function(sampled_params)
                model_result['iteration'] = i
                results.append(model_result)
            except Exception as e:
                logger.warning(f"Model execution failed for iteration {i}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _run_parallel_simulation(self,
                                model_function: Callable,
                                scenario: ScenarioDefinition,
                                n_iterations: int,
                                max_workers: int) -> pd.DataFrame:
        """Run simulation in parallel."""
        results = []
        
        def run_single_iteration(iteration: int) -> Dict[str, Any]:
            sampled_params = self._sample_scenario_parameters(scenario, iteration=iteration)
            model_result = model_function(sampled_params)
            model_result['iteration'] = iteration
            return model_result
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all iterations
            futures = {
                executor.submit(run_single_iteration, i): i 
                for i in range(n_iterations)
            }
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    iteration = futures[future]
                    logger.warning(f"Parallel execution failed for iteration {iteration}: {e}")
        
        return pd.DataFrame(results)
    
    def _sample_scenario_parameters(self, 
                                   scenario: ScenarioDefinition,
                                   iteration: int) -> Dict[str, Any]:
        """Sample parameters for a specific scenario iteration."""
        sampled_params = {}
        
        # Use iteration-specific seed for reproducibility
        local_seed = self.random_seed + iteration
        
        for param_name, distribution in scenario.variable_distributions.items():
            # Ensure seed is within valid range
            param_seed = (local_seed + abs(hash(param_name))) % (2**31)
            sample_value = distribution.sample(size=1, random_state=param_seed)[0]
            sampled_params[param_name] = sample_value
        
        # Add scenario conditions if any
        if scenario.conditions:
            sampled_params.update(scenario.conditions)
        
        return sampled_params
    
    def calculate_risk_metrics(self, 
                              results: pd.DataFrame,
                              target_column: str) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            results: Simulation results DataFrame
            target_column: Column to analyze for risk metrics
            
        Returns:
            Dictionary of risk metrics
        """
        if target_column not in results.columns:
            raise ValueError(f"Column {target_column} not found in results")
        
        values = results[target_column].dropna()
        
        metrics = {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'median': values.median(),
            'skewness': stats.skew(values),
            'kurtosis': stats.kurtosis(values),
        }
        
        # Value at Risk (VaR) at different confidence levels
        for confidence in self.var_confidence_levels:
            var_value = np.percentile(values, (1 - confidence) * 100)
            metrics[f'var_{int(confidence*100)}'] = var_value
        
        # Expected Shortfall (Conditional VaR)
        var_threshold = np.percentile(values, self.expected_shortfall_alpha * 100)
        expected_shortfall = values[values <= var_threshold].mean()
        metrics['expected_shortfall'] = expected_shortfall
        
        # Probability of loss (values below zero)
        prob_loss = (values < 0).mean()
        metrics['probability_of_loss'] = prob_loss
        
        return metrics
    
    def sensitivity_analysis(self,
                           results: pd.DataFrame,
                           target_column: str,
                           input_columns: List[str]) -> pd.DataFrame:
        """
        Perform sensitivity analysis using correlation coefficients.
        
        Args:
            results: Simulation results DataFrame
            target_column: Target output variable
            input_columns: Input variables to analyze
            
        Returns:
            DataFrame with sensitivity metrics
        """
        sensitivity_data = []
        
        for input_col in input_columns:
            if input_col in results.columns:
                # Pearson correlation
                pearson_corr = results[target_column].corr(results[input_col])
                
                # Spearman rank correlation (for non-linear relationships)
                spearman_corr = results[target_column].corr(results[input_col], method='spearman')
                
                # Partial correlation (controlling for other variables)
                other_vars = [col for col in input_columns if col != input_col and col in results.columns]
                if other_vars:
                    # Simple implementation - could be enhanced with proper partial correlation
                    partial_corr = self._calculate_partial_correlation(
                        results, target_column, input_col, other_vars
                    )
                else:
                    partial_corr = pearson_corr
                
                sensitivity_data.append({
                    'input_variable': input_col,
                    'pearson_correlation': pearson_corr,
                    'spearman_correlation': spearman_corr,
                    'partial_correlation': partial_corr,
                    'absolute_correlation': abs(pearson_corr)
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        sensitivity_df = sensitivity_df.sort_values('absolute_correlation', ascending=False)
        
        return sensitivity_df
    
    def _calculate_partial_correlation(self,
                                     data: pd.DataFrame,
                                     target: str,
                                     variable: str,
                                     control_vars: List[str]) -> float:
        """Calculate partial correlation coefficient."""
        try:
            from sklearn.linear_model import LinearRegression
            
            # Regress target on control variables
            X_control = data[control_vars].values
            y_target = data[target].values
            y_variable = data[variable].values
            
            reg_target = LinearRegression().fit(X_control, y_target)
            reg_variable = LinearRegression().fit(X_control, y_variable)
            
            # Get residuals
            target_residuals = y_target - reg_target.predict(X_control)
            variable_residuals = y_variable - reg_variable.predict(X_control)
            
            # Correlation of residuals
            return np.corrcoef(target_residuals, variable_residuals)[0, 1]
            
        except Exception:
            # Fallback to simple correlation if sklearn not available
            return data[target].corr(data[variable])
    
    def generate_scenario_report(self, 
                               results: pd.DataFrame,
                               target_columns: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive scenario analysis report.
        
        Args:
            results: Simulation results DataFrame
            target_columns: List of columns to analyze
            
        Returns:
            Dictionary containing scenario analysis report
        """
        report = {
            'summary': {
                'total_iterations': len(results),
                'scenarios': results['scenario'].unique().tolist() if 'scenario' in results.columns else ['default'],
                'target_variables': target_columns
            },
            'risk_metrics': {},
            'scenario_comparison': {},
            'sensitivity_analysis': {}
        }
        
        # Calculate risk metrics for each target variable
        for target_col in target_columns:
            if target_col in results.columns:
                report['risk_metrics'][target_col] = self.calculate_risk_metrics(results, target_col)
        
        # Scenario comparison if multiple scenarios exist
        if 'scenario' in results.columns and len(results['scenario'].unique()) > 1:
            for target_col in target_columns:
                if target_col in results.columns:
                    scenario_stats = results.groupby('scenario')[target_col].agg([
                        'mean', 'std', 'min', 'max', 'median'
                    ]).round(4)
                    report['scenario_comparison'][target_col] = scenario_stats.to_dict('index')
        
        # Sensitivity analysis
        input_columns = [col for col in results.columns 
                        if col not in target_columns + ['iteration', 'scenario', 'scenario_probability']]
        
        for target_col in target_columns:
            if target_col in results.columns and input_columns:
                sensitivity = self.sensitivity_analysis(results, target_col, input_columns)
                report['sensitivity_analysis'][target_col] = sensitivity.to_dict('records')
        
        return report 