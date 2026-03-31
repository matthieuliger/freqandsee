# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**freqandsee** is a signal processing library for analyzing and manipulating time-series data, built on NumPy, SciPy, and Astropy units.

## Build & Development Commands

```bash
poetry install                                    # Install dependencies
poetry run pytest                                 # Run tests
poetry run pytest tests/test_file.py::test_name   # Run a single test
poetry run mypy src                               # Type checking
poetry run isort src                              # Sort imports
poetry run coverage run --source=src -m pytest    # Tests with coverage
poetry build                                      # Build package
```

## Architecture

All core code lives in `src/freqandsee/freqandsee.py` (~1500 lines).

### Signal Class Hierarchy

- **FlexibleSignal** — Base class for x-y data with astropy units and metadata
  - **Signal** — Strict time-series: uniformly spaced, monotonically increasing x values
    - **DatedSignal**, **ImpulseResponse**, **PSDSignal** — Specialized signal types
    - **SineSignal**, **WhiteNoise**, **FlickerNoise**, **OneSignal** — Signal generators
    - **FrequencyResponseSignal** — Frequency domain representation

### Filter Hierarchy

- **Filter** (abstract base)
  - **LinearFilter** — General linear filtering with b,a coefficients
    - **MovingAverageFilter**, **FBMFlickerFilter**, **BarnesFilter**, **AllanFilter**, **PVarFilter**, **SimpleIntegrator**
  - **FilterBank** — Applies multiple filters collectively

### Key Design Patterns

- All signals carry astropy units on both axes; use `dimensionless()` to strip units when needed for raw numpy operations.
- **SignalBundle** groups multiple signals for collective plotting and analysis.
- Pandas integration supports DataFrame conversion and CSV I/O with unit/metadata preservation.

## Dependencies

Core: numpy, scipy, astropy (units), matplotlib, pandas, seaborn. Custom `pretty-logger` from GitHub for logging.
