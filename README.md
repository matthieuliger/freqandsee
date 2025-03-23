# freqandsee

This repository provides a high-level signal processing library called
`freqandsee`, designed for analyzing and manipulating time-series data. The
library is built on top of `numpy` and `astropy`, providing a flexible
framework for handling signals with various characteristics, including noise
types and filtering techniques. `astropy` is used for unit handling.

## Files

### `freqandsee.py`

Location: freqandsee.py

#### Description

#### Core Functionality

Defines the primary classes and functions for signal processing.

#### Key Components

FlexibleSignal & Signal:
Base classes that represent time-series signals with x (typically time) and y (data) components. They handle units, metadata, plotting, and basic statistical calculations (e.g., RMS, variance).

Signal Subclasses:
Specialized signal types like WhiteNoise, FlickerNoise, and others generate or manipulate specific types of signals.

Filters:
Implements several filtering classes such as LinearFilter, MovingAverageFilter, AllanFilter, BarnesFilter, and FilterBank. These classes enable filtering operations, frequency response analysis, and conversion between continuous and discrete representations (e.g., bilinear transform).

#### Utility Functions

Contains helper functions (e.g., bilinear transformation, Hollos method, and unit handling) that support signal processing and ensure proper unit management.

## Demo Notebooks

### Basic Demo Notebook

[Notebook](demo_notebooks/basic_demo.ipynb)

#### Description

Provides a basic example of how to use the freqandsee library by:

Generating and plotting a sine wave signal.
Instantiating a MovingAverageFilter and examining its frequency response.
Displaying key properties of the signals (like RMS) and analyzing them.

Additional Features:
Explores other aspects of the library, such as creating white noise and possibly flicker noise signals, and performing operations like FFT and PSD estimation on the signals.

### Filtering Demo Notebook

[Notebook](demo_notebooks/filtering_demo.ipynb)

#### Description

Low-Pass Filter Implementation:
Shows how to design a low-pass filter with a cutoff frequency (e.g., 5 kHz) using concepts like the bilinear transform.

Signal Processing:
A sine wave signal is generated and filtered using the designed LinearFilter. The notebook:

Plots the frequency response of the filter.
Filters the signal.
Displays both the original and filtered signals using a SignalBundle for easy comparison.

### Allan Variance White Noise Demo Notebook

[Notebook](demo_notebooks/allan_variance_white_noise_demo.ipynb)

Description:

White Noise Generation:
Demonstrates creating a white noise signal with a specified power spectral density (PSD).

Allan Filter Usage:
Applies an AllanFilter to the white noise signal. This example shows how filtering can be used to compute Allan variance, which is inversely proportional to the moving average length in the filter.

Filter Bank:
A filter bank composed of Allan filters of increasing lengths is created. The notebook filters the white noise signal through this bank and obtains the RMS values of the filtered signals to analyze noise characteristics.

### Import Export Demo Notebook

[Notebook](demo_notebook/import_export_demo.ipynb)

#### Description

Signal Creation:
Creates a sine wave signal that simulates sensor data (e.g., "Aether sensor") with astropy units.

Exporting to CSV:
Illustrates how to export a signal to CSV while including a header with metadata such as units and signal names.

Importing Signals:
Demonstrates importing signal data back from CSV files in three different scenarios:

With a header that includes units.
Without a header (resulting in a unitless signal).
Without a header but with user-specified units during import.
