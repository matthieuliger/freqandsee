import sys
from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from math import pi as π

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from freqandsee.freqandsee import (
    Signal,
    FlexibleSignal,
    WhiteNoise,
    SineSignal,
    LinearFilter,
    MovingAverageFilter,
    AllanFilter,
    FilterBank,
    SignalBundle,
    PSDSignal,
    FrequencyResponseSignal,
    dimensionless,
    bilinear_transform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_signal(n=1000, fs_val=1000.0, y_unit=u.volt):
    """Create a simple sine signal for testing."""
    fs = fs_val * u.Hz
    t = np.arange(n) / fs
    phase_values = (2 * π * 50 * u.Hz * t).to(u.dimensionless_unscaled).value
    y = np.sin(phase_values) * y_unit
    return Signal(x=t, y=y, name="test_sine")


def make_dc_signal(n=100, fs_val=1000.0, value=1.0):
    """Create a constant-value signal."""
    fs = fs_val * u.Hz
    t = np.arange(n) / fs
    y = np.ones(n) * value * u.volt
    return Signal(x=t, y=y, name="dc")


# ---------------------------------------------------------------------------
# Signal creation
# ---------------------------------------------------------------------------

class TestSignalCreation:
    def test_basic_creation(self):
        s = make_signal()
        assert s.n == 1000
        assert s.x_unit == u.s
        assert s.y_unit == u.volt

    def test_sampling_frequency(self):
        s = make_signal(fs_val=500.0)
        assert np.isclose(dimensionless(s.fs * u.s), 500.0)

    def test_non_uniform_raises(self):
        t = np.array([0.0, 1.0, 3.0]) * u.s
        y = np.array([1.0, 2.0, 3.0]) * u.volt
        with pytest.raises(ValueError):
            Signal(x=t, y=y)

    def test_length_mismatch_raises(self):
        t = np.arange(10) * u.s
        y = np.ones(5) * u.volt
        with pytest.raises(ValueError):
            Signal(x=t, y=y)


# ---------------------------------------------------------------------------
# Signal arithmetic
# ---------------------------------------------------------------------------

class TestSignalArithmetic:
    def test_add(self):
        s1 = make_dc_signal(value=1.0)
        s2 = make_dc_signal(value=2.0)
        result = s1 + s2
        assert np.allclose(result.y.value, 3.0)

    def test_sub(self):
        s1 = make_dc_signal(value=5.0)
        s2 = make_dc_signal(value=2.0)
        result = s1 - s2
        assert np.allclose(result.y.value, 3.0)

    def test_add_mismatched_x_raises(self):
        s1 = make_dc_signal(n=100)
        t2 = np.arange(100) * 0.002 * u.s  # different spacing
        y2 = np.ones(100) * u.volt
        s2 = Signal(x=t2, y=y2)
        with pytest.raises(ValueError):
            s1 + s2

    def test_sub_mismatched_x_raises(self):
        s1 = make_dc_signal(n=100)
        t2 = np.arange(100) * 0.002 * u.s  # different spacing
        y2 = np.ones(100) * u.volt
        s2 = Signal(x=t2, y=y2)
        with pytest.raises(ValueError):
            s1 - s2

    def test_mul_scalar(self):
        s = make_dc_signal(value=2.0)
        result = s * 3
        assert np.allclose(result.y.value, 6.0)

    def test_div_scalar(self):
        s = make_dc_signal(value=6.0)
        result = s / 2
        assert np.allclose(result.y.value, 3.0)


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

class TestFilterApplication:
    def test_signal_times_filter(self):
        s = make_signal()
        f = MovingAverageFilter(n=5, name="MA5")
        result = s * f
        assert isinstance(result, Signal)
        assert result.n == s.n

    def test_filter_times_signal_rmul(self):
        s = make_signal()
        f = MovingAverageFilter(n=5, name="MA5")
        result = f * s
        assert isinstance(result, Signal)
        assert result.n == s.n

    def test_filter_apply_method(self):
        s = make_signal()
        f = MovingAverageFilter(n=5, name="MA5")
        result = f.apply(s)
        assert isinstance(result, Signal)

    def test_signal_times_filterbank(self):
        s = make_signal()
        fb = FilterBank(
            filters=[MovingAverageFilter(n=3), MovingAverageFilter(n=5)]
        )
        result = s * fb
        assert isinstance(result, SignalBundle)
        assert len(result) == 2

    def test_moving_average_smooths(self):
        """A moving average of a constant signal should stay constant."""
        s = make_dc_signal(value=4.0, n=200)
        f = MovingAverageFilter(n=10, name="MA10")
        result = s * f
        # After initial transient, values should be ~4.0
        assert np.allclose(result.y.value[10:], 4.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Frequency response and Bode
# ---------------------------------------------------------------------------

class TestFrequencyResponse:
    def test_frequency_response(self):
        f = MovingAverageFilter(n=8, name="MA8")
        fs = 1000 * u.Hz
        fr = f.frequency_response(fs=fs)
        assert isinstance(fr, FrequencyResponseSignal)
        assert fr.n > 0

    def test_bode(self):
        f = MovingAverageFilter(n=8, name="MA8")
        fs = 1000 * u.Hz
        mag, phase = f.bode(fs=fs)
        assert isinstance(mag, FrequencyResponseSignal)
        assert isinstance(phase, FrequencyResponseSignal)
        assert mag.n == phase.n
        assert phase.y_unit == u.deg

    def test_bode_dc_gain(self):
        """At DC (frequency=0), a moving average filter has unity gain."""
        f = MovingAverageFilter(n=8, name="MA8")
        fs = 1000 * u.Hz
        mag, phase = f.bode(fs=fs)
        # First element is DC
        assert np.isclose(mag.y.value[0], 1.0, atol=1e-10)
        assert np.isclose(phase.y.value[0], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FlexibleSignal.to_signal (interpolation)
# ---------------------------------------------------------------------------

class TestToSignal:
    def test_irregular_to_uniform(self):
        x = np.array([0.0, 0.3, 0.5, 0.8, 1.0]) * u.s
        y = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) * u.volt
        fs = FlexibleSignal(x=x, y=y, name="irregular")
        s = fs.to_signal(fs=10 * u.Hz)
        assert isinstance(s, Signal)
        # Should have uniform spacing
        diffs = np.diff(s.x.value)
        assert np.allclose(diffs, diffs[0], atol=1e-10)

    def test_interpolation_preserves_units(self):
        x = np.array([0.0, 1.0, 2.0, 3.0]) * u.s
        y = np.array([0.0, 1.0, 0.0, -1.0]) * u.volt
        fs = FlexibleSignal(x=x, y=y)
        s = fs.to_signal(fs=4 * u.Hz)
        assert s.x_unit == u.s
        assert s.y_unit == u.volt

    def test_cubic_interpolation(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * u.s
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0]) * u.volt
        fs = FlexibleSignal(x=x, y=y)
        s = fs.to_signal(fs=2 * u.Hz, kind="cubic")
        assert s.n > len(x)

    def test_gap_filling(self):
        """Data with a gap in the middle gets interpolated over."""
        x = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]) * u.s
        y = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]) * u.volt
        fs = FlexibleSignal(x=x, y=y, name="gappy")
        s = fs.to_signal(fs=10 * u.Hz)
        # The interpolated value at 0.5s should be ~0.5 (linear interp)
        idx_half = np.argmin(np.abs(s.x.value - 0.5))
        assert np.isclose(s.y.value[idx_half], 0.5, atol=0.05)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

class TestResample:
    def test_resample_upsample(self):
        s = make_signal(n=100, fs_val=1000.0)
        new_fs = 2000 * u.Hz
        resampled = s.resample(new_fs)
        assert resampled.n == 200

    def test_resample_downsample(self):
        s = make_signal(n=200, fs_val=1000.0)
        new_fs = 500 * u.Hz
        resampled = s.resample(new_fs)
        assert resampled.n == 100

    def test_resample_preserves_units(self):
        s = make_signal()
        new_fs = 2000 * u.Hz
        resampled = s.resample(new_fs)
        assert resampled.y_unit == s.y_unit
        assert resampled.x_unit == s.x_unit


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

class TestWindow:
    def test_hann_window(self):
        s = make_dc_signal(value=1.0, n=100)
        windowed = s.window("hann")
        assert windowed.n == s.n
        # Hann window is zero at the first sample
        assert np.isclose(windowed.y.value[0], 0.0, atol=1e-10)
        # Middle values are attenuated but nonzero
        assert windowed.y.value[50] > 0.5

    def test_blackman_window(self):
        s = make_dc_signal(value=1.0, n=100)
        windowed = s.window("blackman")
        assert windowed.n == s.n

    def test_window_preserves_units(self):
        s = make_signal()
        windowed = s.window("hamming")
        assert windowed.y_unit == s.y_unit

    def test_invalid_window_raises(self):
        s = make_signal()
        with pytest.raises(ValueError):
            s.window("not_a_real_window")


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------

class TestCSVRoundTrip:
    def test_flexible_signal_csv(self, tmp_path):
        s = make_dc_signal(n=50)
        csv_path = tmp_path / "test_signal.csv"
        s.to_csv(csv_path)
        loaded = FlexibleSignal.from_csv(csv_path)
        assert np.allclose(loaded.y.value, s.y.value, atol=1e-10)

    def test_signal_csv(self, tmp_path):
        s = make_dc_signal(n=50)
        csv_path = tmp_path / "test_signal.csv"
        s.to_csv(csv_path)
        loaded = Signal.from_csv(csv_path)
        assert isinstance(loaded, Signal)
        assert np.allclose(loaded.y.value, s.y.value, atol=1e-10)


# ---------------------------------------------------------------------------
# Unit propagation
# ---------------------------------------------------------------------------

class TestUnitPropagation:
    def test_mul_scalar_preserves_unit(self):
        s = make_dc_signal(value=1.0)
        result = s * 5
        assert result.y_unit == u.volt

    def test_add_preserves_unit(self):
        s1 = make_dc_signal(value=1.0)
        s2 = make_dc_signal(value=2.0)
        result = s1 + s2
        assert result.y_unit == u.volt

    def test_sub_preserves_unit(self):
        s1 = make_dc_signal(value=3.0)
        s2 = make_dc_signal(value=1.0)
        result = s1 - s2
        assert result.y_unit == u.volt

    def test_filter_preserves_unit(self):
        s = make_signal()
        f = MovingAverageFilter(n=5)
        result = s * f
        assert result.y_unit == u.volt


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------

class TestPSD:
    def test_psd_returns_psd_signal(self):
        s = make_signal(n=1024)
        psd = s.psd()
        assert isinstance(psd, PSDSignal)
        assert psd.n > 0


# ---------------------------------------------------------------------------
# SignalBundle
# ---------------------------------------------------------------------------

class TestSignalBundle:
    def test_creation(self):
        s1 = make_signal()
        s2 = make_signal()
        bundle = SignalBundle([s1, s2])
        assert len(bundle) == 2

    def test_iteration(self):
        s1 = make_signal()
        s2 = make_signal()
        bundle = SignalBundle([s1, s2])
        signals = list(bundle)
        assert len(signals) == 2

    def test_bundle_times_filter(self):
        s1 = make_signal()
        s2 = make_signal()
        bundle = SignalBundle([s1, s2])
        f = MovingAverageFilter(n=5)
        result = bundle * f
        assert isinstance(result, SignalBundle)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_dimensionless(self):
        q = 5.0 * u.dimensionless_unscaled
        assert dimensionless(q) == 5.0

    def test_dimensionless_rejects_units(self):
        q = 5.0 * u.meter
        with pytest.raises(ValueError):
            dimensionless(q)

    def test_bilinear_transform(self):
        result = bilinear_transform(-1.0, 10.0)
        expected = -(- (-1.0) - 2 * 10.0) / (- (-1.0) + 2 * 10.0)
        assert np.isclose(result, expected)
