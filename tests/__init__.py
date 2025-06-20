"""
Testing Framework for ChipGeopolitics Simulation Framework

Comprehensive testing suite including unit tests, integration tests,
performance benchmarks, and validation against real-world data.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Test configuration
TEST_DATA_DIR = project_root / 'tests' / 'data'
TEST_OUTPUTS_DIR = project_root / 'tests' / 'outputs'

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUTS_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0" 