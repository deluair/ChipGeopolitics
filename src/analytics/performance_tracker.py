"""
Performance Tracker for ChipGeopolitics Simulation Framework

Comprehensive performance monitoring, profiling, and optimization
system for simulation execution and resource utilization.
"""

import sys
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import json

# Add project root to path for imports
sys.path.append('src')

from config.constants import *

class MetricType(Enum):
    """Types of performance metrics."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    SIMULATION_THROUGHPUT = "simulation_throughput"
    AGENT_PERFORMANCE = "agent_performance"
    CONVERGENCE_RATE = "convergence_rate"
    ERROR_RATE = "error_rate"
    QUALITY_METRICS = "quality_metrics"

class PerformanceLevel(Enum):
    """Performance level classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric definition."""
    metric_id: str
    metric_type: MetricType
    name: str
    description: str
    unit: str
    target_value: Optional[float]
    warning_threshold: Optional[float]
    critical_threshold: Optional[float]
    aggregation_method: str  # "mean", "sum", "max", "min"
    collection_interval: float  # seconds

@dataclass
class PerformanceMeasurement:
    """Single performance measurement."""
    metric_id: str
    timestamp: datetime
    value: float
    context: Dict[str, Any]
    tags: List[str]

@dataclass
class PerformanceReport:
    """Performance analysis report."""
    report_id: str
    period_start: datetime
    period_end: datetime
    summary_stats: Dict[str, Any]
    metric_analysis: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    trend_analysis: Dict[str, Any]
    comparison_data: Optional[Dict[str, Any]]

class PerformanceTracker:
    """
    Comprehensive performance monitoring and analysis system.
    
    Capabilities:
    - Real-time performance monitoring
    - Resource utilization tracking
    - Simulation throughput analysis
    - Bottleneck identification
    - Performance trend analysis
    - Optimization recommendations
    - Comparative performance analysis
    """
    
    def __init__(self):
        # Metric definitions and data
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.measurements: Dict[str, List[PerformanceMeasurement]] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Performance analysis
        self.performance_reports: Dict[str, PerformanceReport] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.collection_interval = 1.0  # seconds
        
        # Initialize performance metrics
        self._initialize_performance_metrics()
        self._initialize_baseline_data()
    
    def _initialize_performance_metrics(self):
        """Initialize standard performance metrics."""
        
        # Execution time metrics
        self.metrics["total_execution_time"] = PerformanceMetric(
            metric_id="total_execution_time",
            metric_type=MetricType.EXECUTION_TIME,
            name="Total Execution Time",
            description="Total time for simulation execution",
            unit="seconds",
            target_value=300.0,  # 5 minutes target
            warning_threshold=600.0,  # 10 minutes warning
            critical_threshold=1800.0,  # 30 minutes critical
            aggregation_method="sum",
            collection_interval=1.0
        )
        
        self.metrics["agent_step_time"] = PerformanceMetric(
            metric_id="agent_step_time",
            metric_type=MetricType.EXECUTION_TIME,
            name="Agent Step Time",
            description="Average time per agent step",
            unit="milliseconds",
            target_value=10.0,
            warning_threshold=50.0,
            critical_threshold=100.0,
            aggregation_method="mean",
            collection_interval=0.1
        )
        
        # Memory usage metrics
        self.metrics["memory_usage"] = PerformanceMetric(
            metric_id="memory_usage",
            metric_type=MetricType.MEMORY_USAGE,
            name="Memory Usage",
            description="Total memory consumption",
            unit="GB",
            target_value=4.0,
            warning_threshold=8.0,
            critical_threshold=16.0,
            aggregation_method="max",
            collection_interval=1.0
        )
        
        self.metrics["memory_growth_rate"] = PerformanceMetric(
            metric_id="memory_growth_rate",
            metric_type=MetricType.MEMORY_USAGE,
            name="Memory Growth Rate",
            description="Rate of memory usage increase",
            unit="MB/minute",
            target_value=10.0,
            warning_threshold=100.0,
            critical_threshold=500.0,
            aggregation_method="mean",
            collection_interval=5.0
        )
        
        # CPU utilization metrics
        self.metrics["cpu_utilization"] = PerformanceMetric(
            metric_id="cpu_utilization",
            metric_type=MetricType.CPU_UTILIZATION,
            name="CPU Utilization",
            description="Overall CPU usage percentage",
            unit="percentage",
            target_value=70.0,
            warning_threshold=85.0,
            critical_threshold=95.0,
            aggregation_method="mean",
            collection_interval=1.0
        )
        
        # Simulation-specific metrics
        self.metrics["simulation_throughput"] = PerformanceMetric(
            metric_id="simulation_throughput",
            metric_type=MetricType.SIMULATION_THROUGHPUT,
            name="Simulation Throughput",
            description="Simulation steps per second",
            unit="steps/second",
            target_value=100.0,
            warning_threshold=50.0,
            critical_threshold=10.0,
            aggregation_method="mean",
            collection_interval=5.0
        )
        
        self.metrics["convergence_rate"] = PerformanceMetric(
            metric_id="convergence_rate",
            metric_type=MetricType.CONVERGENCE_RATE,
            name="Convergence Rate",
            description="Rate of simulation convergence",
            unit="percentage",
            target_value=95.0,
            warning_threshold=85.0,
            critical_threshold=70.0,
            aggregation_method="mean",
            collection_interval=10.0
        )
        
        self.metrics["error_rate"] = PerformanceMetric(
            metric_id="error_rate",
            metric_type=MetricType.ERROR_RATE,
            name="Error Rate",
            description="Percentage of failed operations",
            unit="percentage",
            target_value=0.1,
            warning_threshold=1.0,
            critical_threshold=5.0,
            aggregation_method="mean",
            collection_interval=1.0
        )
    
    def _initialize_baseline_data(self):
        """Initialize baseline performance data for comparison."""
        
        # Baseline metrics from previous runs or industry standards
        self.baseline_metrics = {
            "execution_performance": {
                "total_execution_time": 240.0,  # 4 minutes
                "agent_step_time": 8.5,  # 8.5 ms
                "simulation_throughput": 120.0  # 120 steps/second
            },
            "resource_utilization": {
                "memory_usage": 3.2,  # 3.2 GB
                "cpu_utilization": 65.0,  # 65%
                "memory_growth_rate": 8.0  # 8 MB/minute
            },
            "quality_metrics": {
                "convergence_rate": 97.5,  # 97.5%
                "error_rate": 0.05  # 0.05%
            }
        }
    
    def start_monitoring_session(self, session_id: str, 
                                context: Optional[Dict[str, Any]] = None) -> None:
        """Start a new performance monitoring session."""
        
        session_context = context or {}
        session_context.update({
            "session_id": session_id,
            "start_time": datetime.now(),
            "status": "active"
        })
        
        self.active_sessions[session_id] = session_context
        
        # Initialize measurement storage for this session
        for metric_id in self.metrics:
            if metric_id not in self.measurements:
                self.measurements[metric_id] = []
        
        # Start system monitoring if not already active
        if not self.monitoring_active:
            self.start_system_monitoring()
        
        print(f"Performance monitoring session '{session_id}' started")
    
    def stop_monitoring_session(self, session_id: str) -> Dict[str, Any]:
        """Stop monitoring session and return summary."""
        
        if session_id not in self.active_sessions:
            return {"error": f"Session {session_id} not found"}
        
        session = self.active_sessions[session_id]
        session["end_time"] = datetime.now()
        session["status"] = "completed"
        session["duration"] = (session["end_time"] - session["start_time"]).total_seconds()
        
        # Generate session summary
        summary = self._generate_session_summary(session_id)
        
        # Clean up
        del self.active_sessions[session_id]
        
        # Stop system monitoring if no active sessions
        if not self.active_sessions and self.monitoring_active:
            self.stop_system_monitoring()
        
        print(f"Performance monitoring session '{session_id}' completed")
        return summary
    
    def record_metric(self, metric_id: str, value: float, 
                     context: Optional[Dict[str, Any]] = None,
                     tags: Optional[List[str]] = None) -> None:
        """Record a performance metric measurement."""
        
        if metric_id not in self.metrics:
            print(f"Warning: Unknown metric {metric_id}")
            return
        
        measurement = PerformanceMeasurement(
            metric_id=metric_id,
            timestamp=datetime.now(),
            value=value,
            context=context or {},
            tags=tags or []
        )
        
        if metric_id not in self.measurements:
            self.measurements[metric_id] = []
        
        self.measurements[metric_id].append(measurement)
        
        # Check thresholds
        self._check_performance_thresholds(metric_id, value)
    
    def start_system_monitoring(self) -> None:
        """Start system resource monitoring thread."""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._system_monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        print("System monitoring started")
    
    def stop_system_monitoring(self) -> None:
        """Stop system resource monitoring."""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        print("System monitoring stopped")
    
    def _system_monitoring_loop(self) -> None:
        """Main monitoring loop for system resources."""
        
        last_memory_measurement = None
        
        while self.monitoring_active:
            try:
                # CPU utilization
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.record_metric("cpu_utilization", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_gb = memory.used / (1024**3)
                self.record_metric("memory_usage", memory_gb)
                
                # Memory growth rate
                if last_memory_measurement:
                    time_diff = time.time() - last_memory_measurement["time"]
                    memory_diff = memory_gb - last_memory_measurement["value"]
                    if time_diff > 0:
                        growth_rate = (memory_diff / time_diff) * 60  # MB per minute
                        self.record_metric("memory_growth_rate", growth_rate * 1024)
                
                last_memory_measurement = {"time": time.time(), "value": memory_gb}
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error in system monitoring: {e}")
                time.sleep(1.0)
    
    def _check_performance_thresholds(self, metric_id: str, value: float) -> None:
        """Check if metric value exceeds performance thresholds."""
        
        metric = self.metrics[metric_id]
        
        if metric.critical_threshold and value >= metric.critical_threshold:
            print(f"CRITICAL: {metric.name} = {value:.2f} {metric.unit} "
                  f"(threshold: {metric.critical_threshold})")
        elif metric.warning_threshold and value >= metric.warning_threshold:
            print(f"WARNING: {metric.name} = {value:.2f} {metric.unit} "
                  f"(threshold: {metric.warning_threshold})")
    
    def _generate_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate summary for a monitoring session."""
        
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        session_start = session["start_time"]
        session_end = session.get("end_time", datetime.now())
        
        summary = {
            "session_id": session_id,
            "duration": (session_end - session_start).total_seconds(),
            "metrics_summary": {},
            "performance_level": PerformanceLevel.AVERAGE,
            "key_findings": []
        }
        
        # Calculate summary statistics for each metric
        for metric_id, metric in self.metrics.items():
            measurements = [
                m for m in self.measurements.get(metric_id, [])
                if session_start <= m.timestamp <= session_end
            ]
            
            if measurements:
                values = [m.value for m in measurements]
                summary["metrics_summary"][metric_id] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1] if values else None
                }
        
        # Determine overall performance level
        summary["performance_level"] = self._assess_performance_level(summary["metrics_summary"])
        
        # Generate key findings
        summary["key_findings"] = self._generate_key_findings(summary["metrics_summary"])
        
        return summary
    
    def _assess_performance_level(self, metrics_summary: Dict[str, Any]) -> PerformanceLevel:
        """Assess overall performance level based on metrics."""
        
        performance_scores = []
        
        for metric_id, stats in metrics_summary.items():
            if metric_id not in self.metrics:
                continue
                
            metric = self.metrics[metric_id]
            current_value = stats.get("mean", 0)
            
            # Calculate performance score (0-1 scale)
            if metric.target_value:
                if metric.critical_threshold and current_value >= metric.critical_threshold:
                    score = 0.0
                elif metric.warning_threshold and current_value >= metric.warning_threshold:
                    score = 0.3
                elif current_value <= metric.target_value:
                    score = 1.0
                else:
                    # Linear interpolation between target and warning
                    warning_thresh = metric.warning_threshold or metric.target_value * 2
                    score = max(0.3, 1.0 - (current_value - metric.target_value) / 
                               (warning_thresh - metric.target_value) * 0.7)
            else:
                score = 0.5  # Neutral if no target defined
                
            performance_scores.append(score)
        
        if not performance_scores:
            return PerformanceLevel.AVERAGE
        
        avg_score = np.mean(performance_scores)
        
        if avg_score >= 0.9:
            return PerformanceLevel.EXCELLENT
        elif avg_score >= 0.7:
            return PerformanceLevel.GOOD
        elif avg_score >= 0.5:
            return PerformanceLevel.AVERAGE
        elif avg_score >= 0.3:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _generate_key_findings(self, metrics_summary: Dict[str, Any]) -> List[str]:
        """Generate key performance findings."""
        
        findings = []
        
        for metric_id, stats in metrics_summary.items():
            if metric_id not in self.metrics:
                continue
                
            metric = self.metrics[metric_id]
            current_value = stats.get("mean", 0)
            
            # Compare with baselines
            baseline_category = None
            baseline_value = None
            
            for category, baselines in self.baseline_metrics.items():
                if metric_id in baselines:
                    baseline_category = category
                    baseline_value = baselines[metric_id]
                    break
            
            if baseline_value:
                pct_change = ((current_value - baseline_value) / baseline_value) * 100
                
                if abs(pct_change) > 20:  # Significant change
                    direction = "increased" if pct_change > 0 else "decreased"
                    findings.append(
                        f"{metric.name} {direction} by {abs(pct_change):.1f}% "
                        f"compared to baseline ({current_value:.2f} vs {baseline_value:.2f})"
                    )
            
            # Check threshold violations
            if metric.critical_threshold and current_value >= metric.critical_threshold:
                findings.append(f"{metric.name} exceeded critical threshold "
                              f"({current_value:.2f} >= {metric.critical_threshold})")
            elif metric.warning_threshold and current_value >= metric.warning_threshold:
                findings.append(f"{metric.name} exceeded warning threshold "
                              f"({current_value:.2f} >= {metric.warning_threshold})")
            
            # Check variability
            if stats.get("std", 0) > current_value * 0.3:  # High variability
                findings.append(f"{metric.name} shows high variability "
                              f"(std: {stats['std']:.2f}, cv: {stats['std']/current_value:.2f})")
        
        return findings[:10]  # Limit to top 10 findings
    
    def generate_performance_report(self, start_time: datetime, 
                                  end_time: datetime,
                                  comparison_period: Optional[Tuple[datetime, datetime]] = None) -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        report_id = f"perf_report_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Extract measurements for the period
        period_measurements = {}
        for metric_id, measurements in self.measurements.items():
            period_data = [
                m for m in measurements
                if start_time <= m.timestamp <= end_time
            ]
            period_measurements[metric_id] = period_data
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(period_measurements)
        
        # Perform metric analysis
        metric_analysis = self._analyze_metrics_performance(period_measurements)
        
        # Identify bottlenecks
        bottlenecks = self._identify_performance_bottlenecks(period_measurements)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(metric_analysis, bottlenecks)
        
        # Trend analysis
        trend_analysis = self._perform_trend_analysis(period_measurements)
        
        # Comparison data if provided
        comparison_data = None
        if comparison_period:
            comparison_data = self._generate_comparison_data(
                period_measurements, comparison_period)
        
        report = PerformanceReport(
            report_id=report_id,
            period_start=start_time,
            period_end=end_time,
            summary_stats=summary_stats,
            metric_analysis=metric_analysis,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            trend_analysis=trend_analysis,
            comparison_data=comparison_data
        )
        
        self.performance_reports[report_id] = report
        return report
    
    def _calculate_summary_statistics(self, measurements: Dict[str, List[PerformanceMeasurement]]) -> Dict[str, Any]:
        """Calculate summary statistics for measurements."""
        
        stats = {}
        
        for metric_id, data in measurements.items():
            if not data:
                continue
                
            values = [m.value for m in data]
            
            stats[metric_id] = {
                "count": len(values),
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "q25": np.percentile(values, 25),
                "q75": np.percentile(values, 75),
                "range": np.max(values) - np.min(values),
                "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            }
        
        return stats
    
    def _analyze_metrics_performance(self, measurements: Dict[str, List[PerformanceMeasurement]]) -> Dict[str, Any]:
        """Analyze performance of individual metrics."""
        
        analysis = {}
        
        for metric_id, data in measurements.items():
            if not data or metric_id not in self.metrics:
                continue
                
            metric = self.metrics[metric_id]
            values = [m.value for m in data]
            
            # Performance against targets
            target_performance = None
            if metric.target_value:
                target_performance = {
                    "target_value": metric.target_value,
                    "actual_mean": np.mean(values),
                    "deviation": np.mean(values) - metric.target_value,
                    "deviation_percentage": ((np.mean(values) - metric.target_value) / metric.target_value) * 100,
                    "meets_target": np.mean(values) <= metric.target_value
                }
            
            # Threshold analysis
            threshold_analysis = {
                "warning_violations": 0,
                "critical_violations": 0
            }
            
            if metric.warning_threshold:
                threshold_analysis["warning_violations"] = sum(1 for v in values if v >= metric.warning_threshold)
            if metric.critical_threshold:
                threshold_analysis["critical_violations"] = sum(1 for v in values if v >= metric.critical_threshold)
            
            # Trend analysis
            if len(values) > 1:
                # Simple linear trend
                x = np.arange(len(values))
                slope, intercept = np.polyfit(x, values, 1)
                trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            else:
                slope = 0
                trend = "insufficient_data"
            
            analysis[metric_id] = {
                "metric_name": metric.name,
                "target_performance": target_performance,
                "threshold_analysis": threshold_analysis,
                "trend": {
                    "direction": trend,
                    "slope": slope,
                    "trend_strength": abs(slope) / np.std(values) if np.std(values) != 0 else 0
                },
                "stability": {
                    "coefficient_of_variation": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    "is_stable": (np.std(values) / np.mean(values)) < 0.1 if np.mean(values) != 0 else False
                }
            }
        
        return analysis
    
    def _identify_performance_bottlenecks(self, measurements: Dict[str, List[PerformanceMeasurement]]) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        
        bottlenecks = []
        
        # CPU bottleneck detection
        if "cpu_utilization" in measurements:
            cpu_data = [m.value for m in measurements["cpu_utilization"]]
            if cpu_data and np.mean(cpu_data) > 85:
                bottlenecks.append({
                    "type": "CPU",
                    "severity": "high" if np.mean(cpu_data) > 95 else "medium",
                    "description": f"High CPU utilization detected (avg: {np.mean(cpu_data):.1f}%)",
                    "impact": "Simulation performance degradation",
                    "metric_id": "cpu_utilization"
                })
        
        # Memory bottleneck detection
        if "memory_usage" in measurements:
            memory_data = [m.value for m in measurements["memory_usage"]]
            if memory_data and np.max(memory_data) > 8.0:  # 8GB threshold
                bottlenecks.append({
                    "type": "Memory",
                    "severity": "high" if np.max(memory_data) > 16.0 else "medium",
                    "description": f"High memory usage detected (max: {np.max(memory_data):.1f} GB)",
                    "impact": "Risk of system slowdown or crashes",
                    "metric_id": "memory_usage"
                })
        
        # Performance degradation detection
        if "simulation_throughput" in measurements:
            throughput_data = [m.value for m in measurements["simulation_throughput"]]
            if throughput_data and np.mean(throughput_data) < 50:  # Below 50 steps/second
                bottlenecks.append({
                    "type": "Throughput",
                    "severity": "high" if np.mean(throughput_data) < 10 else "medium",
                    "description": f"Low simulation throughput (avg: {np.mean(throughput_data):.1f} steps/sec)",
                    "impact": "Extended simulation execution time",
                    "metric_id": "simulation_throughput"
                })
        
        # Error rate issues
        if "error_rate" in measurements:
            error_data = [m.value for m in measurements["error_rate"]]
            if error_data and np.mean(error_data) > 1.0:  # Above 1% error rate
                bottlenecks.append({
                    "type": "Quality",
                    "severity": "critical" if np.mean(error_data) > 5.0 else "high",
                    "description": f"High error rate detected (avg: {np.mean(error_data):.2f}%)",
                    "impact": "Simulation reliability and accuracy concerns",
                    "metric_id": "error_rate"
                })
        
        return sorted(bottlenecks, key=lambda x: {"critical": 3, "high": 2, "medium": 1}.get(x["severity"], 0), reverse=True)
    
    def _generate_performance_recommendations(self, metric_analysis: Dict[str, Any], 
                                           bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations."""
        
        recommendations = []
        
        # Address bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "CPU":
                recommendations.append("Consider CPU optimization: reduce algorithm complexity, implement parallelization, or upgrade hardware")
            elif bottleneck["type"] == "Memory":
                recommendations.append("Address memory usage: implement memory pooling, optimize data structures, or increase system RAM")
            elif bottleneck["type"] == "Throughput":
                recommendations.append("Improve throughput: profile code for hotspots, optimize critical paths, consider distributed execution")
            elif bottleneck["type"] == "Quality":
                recommendations.append("Address quality issues: review error handling, improve input validation, enhance debugging capabilities")
        
        # Metric-specific recommendations
        for metric_id, analysis in metric_analysis.items():
            target_perf = analysis.get("target_performance")
            if target_perf and not target_perf.get("meets_target"):
                deviation_pct = abs(target_perf.get("deviation_percentage", 0))
                if deviation_pct > 50:
                    recommendations.append(f"Significant performance gap in {analysis['metric_name']}: "
                                         f"consider architectural improvements")
                elif deviation_pct > 20:
                    recommendations.append(f"Performance tuning needed for {analysis['metric_name']}: "
                                         f"optimize relevant algorithms and configurations")
            
            # Stability recommendations
            stability = analysis.get("stability", {})
            if not stability.get("is_stable"):
                recommendations.append(f"Improve {analysis['metric_name']} stability: "
                                     f"investigate and reduce performance variability")
        
        # General recommendations
        recommendations.extend([
            "Regular performance monitoring and profiling recommended",
            "Consider implementing performance regression tests",
            "Establish performance baselines for future comparisons",
            "Monitor trends for early detection of performance degradation"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _perform_trend_analysis(self, measurements: Dict[str, List[PerformanceMeasurement]]) -> Dict[str, Any]:
        """Perform trend analysis on performance metrics."""
        
        trends = {}
        
        for metric_id, data in measurements.items():
            if len(data) < 5:  # Need minimum data points for trend analysis
                continue
                
            values = [m.value for m in data]
            timestamps = [m.timestamp for m in data]
            
            # Convert timestamps to numeric values for regression
            time_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]
            
            # Linear trend analysis
            if len(values) > 1:
                slope, intercept = np.polyfit(time_numeric, values, 1)
                r_squared = np.corrcoef(time_numeric, values)[0, 1] ** 2
                
                # Classify trend
                if abs(slope) < 0.001:  # Minimal change
                    trend_type = "stable"
                elif slope > 0:
                    trend_type = "increasing"
                else:
                    trend_type = "decreasing"
                
                # Trend strength
                std_dev = np.std(values)
                trend_strength = abs(slope) / std_dev if std_dev != 0 else 0
                
                if trend_strength > 1.0:
                    strength = "strong"
                elif trend_strength > 0.5:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                trends[metric_id] = {
                    "trend_type": trend_type,
                    "slope": slope,
                    "r_squared": r_squared,
                    "trend_strength": strength,
                    "strength_value": trend_strength,
                    "prediction_confidence": "high" if r_squared > 0.8 else "medium" if r_squared > 0.5 else "low"
                }
        
        return trends
    
    def _generate_comparison_data(self, current_measurements: Dict[str, List[PerformanceMeasurement]], 
                                comparison_period: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Generate comparison data between current and previous period."""
        
        start_time, end_time = comparison_period
        
        # Extract comparison period measurements
        comparison_measurements = {}
        for metric_id, measurements in self.measurements.items():
            comparison_data = [
                m for m in measurements
                if start_time <= m.timestamp <= end_time
            ]
            comparison_measurements[metric_id] = comparison_data
        
        comparison = {}
        
        for metric_id in current_measurements.keys():
            current_data = [m.value for m in current_measurements.get(metric_id, [])]
            comparison_data = [m.value for m in comparison_measurements.get(metric_id, [])]
            
            if current_data and comparison_data:
                current_mean = np.mean(current_data)
                comparison_mean = np.mean(comparison_data)
                
                change = current_mean - comparison_mean
                pct_change = (change / comparison_mean) * 100 if comparison_mean != 0 else 0
                
                comparison[metric_id] = {
                    "current_mean": current_mean,
                    "previous_mean": comparison_mean,
                    "absolute_change": change,
                    "percentage_change": pct_change,
                    "improvement": change < 0 if "error" in metric_id or "time" in metric_id else change > 0
                }
        
        return comparison
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance tracking summary."""
        
        return {
            "metrics_defined": len(self.metrics),
            "active_sessions": len(self.active_sessions),
            "total_measurements": sum(len(measurements) for measurements in self.measurements.values()),
            "monitoring_active": self.monitoring_active,
            "reports_generated": len(self.performance_reports),
            "metric_types": list(set(metric.metric_type.value for metric in self.metrics.values())),
            "baseline_categories": list(self.baseline_metrics.keys()),
            "capabilities": [
                "Real-time performance monitoring",
                "Resource utilization tracking", 
                "Bottleneck identification",
                "Trend analysis and forecasting",
                "Performance comparison",
                "Optimization recommendations",
                "Threshold-based alerting"
            ]
        }