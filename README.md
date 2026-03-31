# freqandsee

A high-level signal processing library for analyzing and manipulating
time-series data, built on top of `numpy`, `scipy`, and `astropy` (for unit
handling).

## Installation

Requires Python >= 3.12.

```bash
poetry install
```

## Quick Example

```python
import numpy as np
from astropy import units as u
from math import pi as π
from freqandsee.freqandsee import Signal, MovingAverageFilter, SignalBundle

# Create a sine wave signal
fs = 100 * u.kHz
time = np.arange(0, 101) / fs
fc = 3000 * u.hertz
y = Signal(
    x=time,
    y=np.sin(2 * π * fc * time) * u.volt,
    name="Aether sensor",
    x_name="t",
    y_name="Voltage",
)

# Filter it with a moving average (signal * filter uses operator overloading)
h = MovingAverageFilter(n=8, name="MA")
y_filtered = y * h

# Plot both together
SignalBundle([y, y_filtered]).plot()
```

## Library Overview

All core code lives in `src/freqandsee/freqandsee.py`.

### Signal Class Hierarchy

- **FlexibleSignal** — Base class for x-y data with astropy units and metadata. Handles plotting, statistics (RMS, variance), and unit management.
  - **Signal** — Strict time-series: uniformly spaced, monotonically increasing x values.
    - **DatedSignal** — Signal with date information.
    - **ImpulseResponse** — Impulse response signals.
    - **PSDSignal** — Power Spectral Density signals.
    - **OneSignal** — Unit amplitude signal.
    - **SineSignal** — Sine wave generation.
    - **WhiteNoise** — White noise with specified PSD.
    - **FlickerNoise** — Flicker (pink) noise generation.
    - **FrequencyResponseSignal** — Frequency domain representation.

### Filter Hierarchy

- **Filter** — Abstract base class.
  - **LinearFilter** — General linear filtering with b, a coefficients. Supports frequency response analysis and bilinear transform.
    - **MovingAverageFilter** — Moving average filtering.
    - **FBMFlickerFilter** — Fractional Brownian Motion flicker filter.
    - **BarnesFilter** — Barnes bilinear filter.
    - **AllanFilter** — Allan variance filter.
    - **PVarFilter** — Power variance filter.
    - **SimpleIntegrator** — Integration filter.
  - **FilterBank** — Applies multiple filters collectively.

### Utilities

- **SignalBundle** — Container for multiple signals with collective plotting and analysis.
- **bilinear_transform()** — Bilinear transformation on poles.
- **hollos()** — Transfer function computation using the Hollos method.
- **dimensionless()** — Convert astropy quantities to dimensionless values.

## Demo Notebooks

- [Basic demo](demo_notebooks/basic_demo.ipynb) — Sine wave creation, MovingAverageFilter frequency response, signal properties (RMS), FFT and PSD estimation.
- [Filtering demo](demo_notebooks/filtering_demo.ipynb) — Low-pass filter design with bilinear transform, signal filtering with LinearFilter, before/after comparison via SignalBundle.
- [Allan variance white noise demo](demo_notebooks/allan_variance_white_noise_demo.ipynb) — White noise generation with specified PSD, Allan variance computation via AllanFilter, FilterBank usage for noise characterization.
- [Import/export demo](demo_notebooks/import_export_demo.ipynb) — CSV export with unit/metadata headers, import with and without headers, user-specified units on import.
