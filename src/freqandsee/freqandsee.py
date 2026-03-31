import ast
from pathlib import Path
import pandas as pd
from typing import Union, Optional, List, cast, Callable, Tuple, TypeAlias

from scipy import signal as scp_signal  # type: ignore
import numpy as np
from numpy.typing import NDArray

import numbers
import scipy
import astropy.units as u  # type: ignore
import matplotlib.pyplot as plt
from math import pi as π
from numpy.polynomial.polynomial import polyval
from abc import abstractmethod

from freqandsee.project_logger import logger

try:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )
except Exception:
    pass
AnyNumber: TypeAlias = numbers.Number | float | int
AnyScalar = (numbers.Number, float, int, u.Unit, u.Quantity)

RELATIVE_EQUALITY_THRESHOLD = 1e-11


physical_types_units = {"frequency": u.hertz}


def bilinear_transform(pole, fs):
    """Perform a bilinear transformation on the given pole value.

    Parameters
    ----------
    pole : float or array-like
        The pole value to transform.
    fs : float or astropy.units.Quantity
        The sampling frequency.

    Returns
    -------
    float or array-like
        The transformed pole value.
    """
    return -(-pole - 2 * fs) / (-pole + 2 * fs)


def hollos(n):
    """Compute transfer function coefficients using the Hollos method.

    Parameters
    ----------
    n : int
        Number of poles/zeros to generate.

    Returns
    -------
    b : numpy.ndarray
        Numerator coefficients.
    a : numpy.ndarray
        Denominator coefficients.
    """
    bk = [np.cos(2 * k * np.pi / float(2 * n + 1)) for k in range(1, n + 1)]
    bk = np.asarray(bk)

    zzeros = bk
    zpoles = -bk

    b, a = scp_signal.zpk2tf(zzeros, zpoles, 1)
    return b, a


def dimensionless(x: u.Quantity):
    """Convert an astropy Quantity to a dimensionless numeric value.

    Parameters
    ----------
    x : astropy.units.Quantity
        A Quantity expected to have dimensionless units.

    Returns
    -------
    float or numpy.ndarray
        The numeric value after converting to dimensionless.

    Raises
    ------
    ValueError
        If ``x`` does not have dimensionless units.
    """
    unit = x.unit if x.unit is not None else u.dimensionless_unscaled
    if not unit.physical_type == "dimensionless":
        raise ValueError(
            "The unit of the input must be dimensionless, "
            "got {}".format(unit)
        )
    return x.to(u.dimensionless_unscaled).value


def barnes_bilinear(f0, finf, n, K, fs, gain, fref):
    """Compute the poles and zeros of a Barnes bilinear filter.

    Parameters
    ----------
    f0 : astropy.units.Quantity
        Base frequency.
    finf : astropy.units.Quantity
        Upper limit frequency.
    n : int
        Number of filter stages.
    K : float
        Filter parameter.
    fs : astropy.units.Quantity
        Sampling frequency.
    gain : float
        Desired gain.
    fref : astropy.units.Quantity
        Reference frequency.

    Returns
    -------
    b : numpy.ndarray
        Numerator coefficients.
    a : numpy.ndarray
        Denominator coefficients.
    """
    xs = 1 / fs
    alpha = np.longdouble(-1)
    gamma = 10 ** (
        (2 * np.log10(K) + np.log10(finf / f0)) / (n - 1 - alpha / 2.0)
    )
    delta = gamma ** (-alpha / float(2))

    wl = [(gamma**j) * 2 * np.pi * f0 / float(K) for j in range(n)]
    wl = u.Quantity(wl)

    wh = [delta * (gamma**j) * 2 * np.pi * f0 / float(K) for j in range(n)]
    wh = u.Quantity(wh)

    # get digital roots from the continous expresion
    # numerator and denominator
    roots_a = dimensionless(bilinear_transform(-wl, fs))
    roots_b = dimensionless(bilinear_transform(-wh, fs))

    a = np.poly(roots_a)
    b = np.poly(roots_b) * gain

    # ensure the gain at fref is the passed gain
    z = np.exp(1j * 2 * fref * np.pi * xs)
    polya = polyval(z, a)
    polyb = polyval(z, b)
    b = gain * b / (float(abs(polyb / polya)))

    return b, a


def array_from_list(homogenous_list) -> np.ndarray:
    """Convert a list of astropy Quantities to a numpy array preserving units.

    Parameters
    ----------
    homogenous_list : list of astropy.units.Quantity
        Quantity objects that share the same unit.

    Returns
    -------
    numpy.ndarray
        Array of quantities with the common unit.

    Raises
    ------
    ValueError
        If the elements have differing units.
    """
    units = set([element.unit for element in homogenous_list])
    if len(units) > 1:
        raise ValueError("List must be made of a unique unit")
    else:
        list_unit = units.pop()
    # np.array does not know how to handle astropy quantities
    return (
        np.array([element / list_unit for element in homogenous_list])
        * list_unit
    )


def get_interval(index):
    """Determine the constant interval between successive values in a sequence.

    Parameters
    ----------
    index : array-like or astropy.units.Quantity
        A sequence of values, possibly with units.

    Returns
    -------
    float or astropy.units.Quantity or int
        The constant interval if uniform, otherwise -1.
    """
    if hasattr(index, "unit"):
        index_unit = index.unit
    else:
        index_unit = 1

    if len(index) == 2:
        return index[1] - index[0]

    difference = index.diff()

    if not all(difference > 0):
        return -1

    if not all(
        np.abs((difference - difference[0]) / index_unit)
        < RELATIVE_EQUALITY_THRESHOLD
    ):
        return -1
    else:
        return difference[0]


class FlexibleSignal:
    """A signal with arbitrary x and y data, each carrying astropy units.

    Provides data analysis (RMS, variance, std, mean), CSV I/O, and
    plotting with unit-aware axis scaling.

    Parameters
    ----------
    x : array-like or astropy.units.Quantity
        The x-axis data (e.g. time, frequency).
    y : array-like or astropy.units.Quantity
        The y-axis data (e.g. voltage, amplitude).
    name : str, optional
        Descriptive name for the signal. Auto-generated if not provided.
    x_name : str, optional
        Display name for the x-axis.
    y_name : str, optional
        Display name for the y-axis.

    Attributes
    ----------
    x_unit : astropy.units.Unit
        Unit of the x-axis data.
    y_unit : astropy.units.Unit
        Unit of the y-axis data.
    y_max : astropy.units.Quantity
        Maximum y value.
    y_min : astropy.units.Quantity
        Minimum y value.
    y_mean : astropy.units.Quantity
        Mean y value.
    """

    instance_counter = 0
    to_string_options = {"format": "latex"}

    def __init__(
        self,
        x,
        y,
        name: Optional[str] = None,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
    ):
        self.x = x
        self.y = y
        FlexibleSignal.instance_counter += 1

        if x_name is not None:
            self.x_name = x_name
        else:
            self.x_name = f"x_{self.instance_counter}"

        if y_name is not None:
            self.y_name = y_name
        else:
            self.y_name = f"y_{self.instance_counter}"

        if name is not None:
            self.name = name
        else:
            self.name = f"Flexiblesignal_{self.instance_counter}"

        self.x_unit, self.y_unit = self.get_units(x, y)
        self.y_max = np.max(self.y)
        self.y_min = np.min(self.y)
        self.y_mean = np.mean(self.y)

    def to_df(self) -> pd.DataFrame:
        """Return a DataFrame with the signal data (unitless).

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns for x and y data, units stripped.
        """
        return pd.DataFrame(
            {self.x_name: self.x.value, self.y_name: self.y.value}
        )

    @classmethod
    def from_csv(cls, path: Path) -> "FlexibleSignal":
        """Read signal data from a CSV file with a metadata header.

        Parameters
        ----------
        path : pathlib.Path
            Path to the CSV file. Expects a ``# {...}`` header line
            containing x/y names and units.

        Returns
        -------
        FlexibleSignal
            Signal loaded from the CSV file.
        """
        with open(path, "r") as csv_file:
            header = csv_file.readline()
            header_dict = ast.literal_eval(header[2:])

        df = pd.read_csv(path, sep=",", header=0, skiprows=1)
        return FlexibleSignal(
            x=df[header_dict["x"]["name"]].values
            * u.Unit(header_dict["x"]["unit"]),
            y=df[header_dict["y"]["name"]].values
            * u.Unit(header_dict["y"]["unit"]),
            name=path.stem,
            x_name=header_dict["x"]["name"],
            y_name=header_dict["y"]["name"],
        )

    def to_csv(self, path: Union[Path, str], add_comment=True):
        """Export signal data to a CSV file.

        Parameters
        ----------
        path : pathlib.Path or str
            File path for the CSV output.
        add_comment : bool, optional
            If True, write a ``# {...}`` header line with axis names and
            units. Default is True.
        """

        if isinstance(path, str):
            path = Path(path)

        df = self.to_df()
        if add_comment:
            header_dict = {
                "x": {"name": self.x_name, "unit": self.x_unit.to_string()},
                "y": {"name": self.y_name, "unit": self.y_unit.to_string()},
            }
        with open(path, "w") as f:
            f.write(f"# {str(header_dict)}\n")
            df.to_csv(f, index=False, header=True, sep=",", mode="a")

    @property
    def x_unit_latexstring(self):
        """str : LaTeX-formatted string of the x-axis unit."""
        return self.x.unit.to_string(**self.to_string_options)

    @property
    def y_unit_latexstring(self):
        """str : LaTeX-formatted string of the y-axis unit."""
        return self.y.unit.to_string(**self.to_string_options)

    @property
    def x_unit_string(self):
        """str : Unicode string of the x-axis unit."""
        return self.x.unit.to_string(format="unicode")

    @property
    def y_unit_string(self):
        """str : Unicode string of the y-axis unit."""
        return self.y.unit.to_string(format="unicode")

    @property
    def rms(self):
        """astropy.units.Quantity : Root mean square of the y-data."""
        return np.sqrt(np.mean(self.y**2))

    @property
    def var(self):
        """astropy.units.Quantity : Variance of the y-data."""
        return np.var(self.y)

    @property
    def std(self):
        """astropy.units.Quantity : Standard deviation of the y-data."""
        return np.std(self.y)

    @property
    def mean(self):
        """astropy.units.Quantity : Mean of the y-data."""
        return np.mean(self.y)

    @property
    def n(self):
        """int : Number of data points."""
        return len(self.x)

    @property
    def fs(self):
        """astropy.units.Quantity : Sampling frequency."""
        return self._fs

    @property
    def unitless_values(self):
        """numpy.ndarray : The y-data with units stripped."""
        return self.y.value

    @classmethod
    def get_units(
        cls,
        x: u.Quantity,
        y: u.Quantity,
    ) -> Tuple[u.Unit, u.Unit]:
        y0 = y[0]
        if hasattr(y0, "unit"):
            y_unit: u.Unit = y0.unit  # type: ignore
        else:
            y_unit = u.dimensionless_unscaled  # type: ignore

        x0 = x[0]
        if hasattr(x0, "unit"):
            x_unit: u.Unit = x0.unit  # type: ignore
        else:
            x_unit = u.dimensionless_unscaled  # type: ignore

        if len(x) != len(y):
            logger.error(error := "x and y must have same length")
            raise ValueError(error)

        return (x_unit, y_unit)

    @staticmethod
    def _as_unit(display_unit):
        """Extract a Unit from a Quantity (e.g. 1/u.day) if needed."""
        if isinstance(display_unit, u.Quantity):
            return display_unit.unit
        return display_unit

    def x_scale_display(
        self, x_display_unit: u.Unit, x_label: Optional[str] = None
    ):
        if x_label is None:
            x_label = self.x_name
        scaled_x = dimensionless(self.x / x_display_unit)
        x_display_unit = self._as_unit(x_display_unit)

        scaled_unit_string = x_display_unit.to_string(**self.to_string_options)

        if x_display_unit.physical_type != "dimensionless":
            scaled_unit_string = f"({scaled_unit_string})"

        scaled_x_label = f"{x_label}" f" {scaled_unit_string}"

        return (
            scaled_x,
            scaled_x_label,
        )

    def y_scale_display(
        self, y_display_unit: u.Unit, y_label: Optional[str] = None
    ):
        if y_label is None:
            y_label = self.y_name
        scaled_y = dimensionless(self.y / y_display_unit)
        y_display_unit = self._as_unit(y_display_unit)

        scaled_unit_string = y_display_unit.to_string(**self.to_string_options)

        if y_display_unit.physical_type != "dimensionless":
            scaled_unit_string = f"({scaled_unit_string})"
        scaled_y_label = f"{y_label}" f" {scaled_unit_string}"
        return (
            scaled_y,
            scaled_y_label,
        )

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def to_signal(
        self,
        fs: u.Quantity,
        kind: str = "linear",
    ) -> "Signal":
        """Interpolate onto a uniform grid and return a Signal.

        Useful for converting irregularly-spaced or gap-containing data
        into a uniformly-sampled Signal.

        Parameters
        ----------
        fs : astropy.units.Quantity
            Desired sampling frequency.
        kind : str, optional
            Interpolation method passed to ``scipy.interpolate.interp1d``
            (e.g. ``"linear"``, ``"cubic"``). Default is ``"linear"``.

        Returns
        -------
        Signal
            A uniformly-sampled signal.
        """
        from scipy.interpolate import interp1d

        x_vals = self.x.value
        y_vals = self.y.value

        ts = 1 / fs
        new_x_vals = np.arange(x_vals[0], x_vals[-1], dimensionless(ts / self.x_unit))
        interpolator = interp1d(x_vals, y_vals, kind=kind)
        new_y_vals = interpolator(new_x_vals)

        new_x = new_x_vals * self.x_unit
        new_y = new_y_vals * self.y_unit

        return Signal(
            x=new_x,
            y=new_y,
            name=f"{self.name} (interpolated {fs})",
            x_name=self.x_name,
            y_name=self.y_name,
        )

    def loglogplot(self, **kwargs):
        """Plot the signal with logarithmic x and y axes.

        Parameters
        ----------
        **kwargs
            Passed to :meth:`plot`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        ax = self.plot(**kwargs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax

    def plot(
        self,
        x_display_unit: Optional[u.Unit] = None,
        y_display_unit: Optional[u.Unit] = None,
        **kwargs,
    ):
        """Plot the signal with optional unit scaling.

        Parameters
        ----------
        x_display_unit : astropy.units.Unit, optional
            Unit to scale the x-axis to. Defaults to the signal's x unit.
        y_display_unit : astropy.units.Unit, optional
            Unit to scale the y-axis to. Defaults to the signal's y unit.
        **kwargs
            Additional keyword arguments: ``ax`` (existing axes),
            ``plot_type`` (``"line"`` or ``"scatter"``), ``title``,
            ``x_label``, ``y_label``, ``suptitle``.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = plt.subplot()

        if x_display_unit is None:
            x_unit: u.Unit = cast(u.Unit, self.x.unit)
            x_display_unit = x_unit

        if y_display_unit is None:
            y_unit: u.Unit = cast(u.Unit, self.y.unit)
            y_display_unit = y_unit

        scaled_x, scaled_x_label = self.x_scale_display(x_display_unit)
        scaled_y, scaled_y_label = self.y_scale_display(y_display_unit)

        ax = self.plot_bare(
            scaled_x=scaled_x,
            scaled_y=scaled_y,
            x_display_unit=x_display_unit,
            y_display_unit=y_display_unit,
            **kwargs,
        )

        # Set custom labels if provided in kwargs
        if "x_label" in kwargs:
            ax.set_xlabel(kwargs.pop("x_label"))
        else:
            ax.set_xlabel(scaled_x_label)

        if "y_label" in kwargs:
            ax.set_ylabel(kwargs.pop("y_label"))
        else:
            ax.set_ylabel(scaled_y_label)

        if "title" in kwargs:
            ax.set_title(kwargs.pop("title"))
        else:
            ax.set_title(self.name)

        if "suptitle" in kwargs:
            ax.fig.suptitle(kwargs.pop("suptitle"))

        ax.grid(which="both", axis="both")

        return ax

    def plot_bare(
        self,
        scaled_x: np.ndarray,
        scaled_y: np.ndarray,
        x_display_unit: Optional[u.Unit] = None,
        y_display_unit: Optional[u.Unit] = None,
        **kwargs,
    ):
        """Plot pre-scaled data without axis labels or titles.

        Parameters
        ----------
        scaled_x : numpy.ndarray
            Unitless, pre-scaled x data.
        scaled_y : numpy.ndarray
            Unitless, pre-scaled y data.
        x_display_unit : astropy.units.Unit, optional
            Display unit for x (used internally).
        y_display_unit : astropy.units.Unit, optional
            Display unit for y (used internally).
        **kwargs
            ``ax``, ``plot_type`` (``"line"`` or ``"scatter"``), and any
            matplotlib plot kwargs.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            ax = plt.subplot()

        if x_display_unit is None:
            x_unit: u.Unit = cast(u.Unit, self.x.unit)
            x_display_unit = x_unit

        if y_display_unit is None:
            y_unit: u.Unit = cast(u.Unit, self.y.unit)
            y_display_unit = y_unit

        if "plot_type" in kwargs:
            plot_type = kwargs.pop("plot_type")
            if plot_type == "line":
                kwargs.pop("plot_type", "line")
                ax.plot(scaled_x, scaled_y, **kwargs)
            elif plot_type == "scatter":
                ax.scatter(scaled_x, scaled_y, **kwargs)
            else:
                logger.error(
                    error := f"plot_type {kwargs['plot_type']} not recognized"
                )
                raise ValueError(error)
        else:
            ax.plot(scaled_x, scaled_y, **kwargs)
        return ax


class Signal(FlexibleSignal):
    """A uniformly-sampled time-series signal.

    The x-axis must be monotonically increasing with constant spacing,
    which is required for FFT, PSD, convolution, and filtering operations.

    Supports arithmetic: ``signal + signal``, ``signal - signal``,
    ``signal * scalar``, ``signal * filter``, ``signal * filterbank``,
    ``signal / scalar``.

    Parameters
    ----------
    x : astropy.units.Quantity
        Uniformly spaced, monotonically increasing time values.
    y : astropy.units.Quantity
        Signal values corresponding to each time point.
    name : str, optional
        Signal name. Auto-generated if not provided.
    x_name : str, optional
        Display name for the x-axis.
    y_name : str, optional
        Display name for the y-axis.

    Attributes
    ----------
    begin : astropy.units.Quantity
        First x value.
    end : astropy.units.Quantity
        Last x value.
    xs : astropy.units.Quantity
        Constant interval between successive x values.
    """

    instance_counter = 0

    @staticmethod
    def from_timed_dataframe(df: pd.DataFrame, y_unit: u.Unit):
        """Create a Signal from a pandas DataFrame with a DatetimeIndex.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with a DatetimeIndex.
        y_unit : astropy.units.Unit
            Unit to assign to the y values.

        Returns
        -------
        Signal or None
            Signal with x in days since the first timestamp, or None if
            the index is not a DatetimeIndex.
        """
        if isinstance(df.index, pd.DatetimeIndex):
            x = (
                (df.index - df.index[0]) / pd.Timedelta(days=1)
            ).values * u.day
            y = df.values * y_unit
            return Signal(x, y)

    def __init__(
        self,
        x: u.Quantity,
        y: u.Quantity,
        name: Optional[str] = None,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
    ):

        self.begin = x.item(0)
        self.end = x.item(-1)
        self._length = self.end - self.begin

        self.x_unit, self.y_unit = FlexibleSignal.get_units(x, y)

        if (x_interval := get_interval(x)) == -1:
            logger.error(
                error := "Index must be monotonically increasing,"
                " by a constant increment"
            )
            raise ValueError(error)

        self.xs = x_interval
        self._fs = 1 / self.xs

        Signal.instance_counter += 1

        if not name:
            name = f"Signal_{Signal.instance_counter}"
        self._name = name

        if x_name is not None:
            self.x_name = x_name
        else:
            self.x_name = f"x({self.name})"

        if y_name is not None:
            self.y_name = y_name
        else:
            self.y_name = f"y({self.name})"

        self.x = x
        self.y = y
        self.y_max = np.max(self.y)
        self.y_min = np.min(self.y)
        self.y_mean = np.mean(self.y)

    @property
    def length(self):
        return self._length

    @classmethod
    def from_csv(
        cls,
        path: Path,
        name: Optional[str] = None,
        x_unit: Optional[u.Unit] = u.dimensionless_unscaled,
        y_unit: Optional[u.Unit] = u.dimensionless_unscaled,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        skiprows: int = 1,
    ) -> "Signal":
        with open(path, "r") as csv_file:
            header = csv_file.readline()
            try:
                header_dict = ast.literal_eval(header[2:])
                df = pd.read_csv(path, sep=",", header=0, skiprows=skiprows)

                return Signal(
                    x=df[header_dict["x"]["name"]].values
                    * u.Unit(header_dict["x"]["unit"]),
                    y=df[header_dict["y"]["name"]].values
                    * u.Unit(header_dict["y"]["unit"]),
                    name=path.stem,
                    x_name=header_dict["x"]["name"],
                    y_name=header_dict["y"]["name"],
                )
            except (ValueError, KeyError, SyntaxError):
                header_dict = {}
                df = pd.read_csv(path, sep=",", header=0, skiprows=skiprows)
                if isinstance(df.columns[0], str):
                    x_name = df.columns[0]
                else:
                    x_name = "x"
                if isinstance(df.columns[1], str):
                    y_name = df.columns[1]
                else:
                    y_name = "y"

                return Signal(
                    x=df.iloc[:, 0].to_numpy() * x_unit,
                    y=df.iloc[:, 1].to_numpy() * y_unit,
                    x_name=x_name,
                    y_name=y_name,
                    name=path.stem,
                )

    def __getitem__(self, index):
        return self.__class__(x=self.x[index], y=self.y[index])

    def __repr__(self):
        return (
            f"Signal name: '{self.name}' "
            f"{self.y_unit_string}"
            f"({self.x_unit_string}). n={self.n:.2e} "
            f"sampling points, fs={self.fs}, "
            f"RMS={self.rms:.2e}."
        )

    @classmethod
    def get_xs(cls, index):
        """Verify and return the uniform interval of an index.

        Parameters
        ----------
        index : array-like
            Sequence of x values.

        Returns
        -------
        float or astropy.units.Quantity
            The constant interval.

        Raises
        ------
        ValueError
            If the index is not uniformly spaced.
        """
        if (result := get_interval(index)) == -1:
            logger.error(
                error := "Index must be monotonically increasing,"
                " by a constant increment"
            )
            raise ValueError(error)
        else:
            return result

    @property
    def fftfreq(self):
        """numpy.ndarray : FFT frequency bins (normalized, fftshift order)."""
        if self.n % 2 == 0:
            return np.hstack(
                [
                    ((np.arange(self.n / 2, self.n, 1) / self.n) - 1),
                    (np.arange(0, self.n / 2, 1) / self.n),
                ]
            )
        else:
            return np.hstack(
                [
                    (np.arange((self.n - 1) / 2 + 1, self.n, 1) / self.n - 1),
                    (np.arange(0, (self.n - 1) / 2 + 1, 1)) / self.n,
                ]
            )

    def apply(
        self, applied_function: Callable
    ) -> Union[AnyNumber, u.Quantity]:
        """Apply a function to the unitless y-data values.

        Parameters
        ----------
        applied_function : callable
            Function to apply to ``self.y.value``.

        Returns
        -------
        result
            The return value of ``applied_function``.
        """
        return applied_function(self.y.value)

    def fft(self):
        """Compute the FFT of the signal.

        Returns
        -------
        FlexibleSignal
            Signal in the frequency domain (fftshift applied).
        """
        fft_values = scipy.fft.fftshift(scipy.fft.fft(self.y.value))
        frequencies = self.fftfreq * self.fs
        return FlexibleSignal(
            x=frequencies,
            y=fft_values * self.y_unit * self.xs,
            name=f"FFT of {self.name}",
            x_name="Frequency",
            y_name=f"FFT({self.name})",
        )

    def __mul__(self, other):
        if isinstance(other, AnyScalar):
            return Signal(y=self.y * other, x=self.x)

        elif isinstance(other, LinearFilter):
            y = (
                scp_signal.lfilter(b=other.b, a=other.a, x=self.y.value)
                * self.y_unit
            )
            name = f"{self.name} * {other.name}"
            return Signal(x=self.x, y=y, name=name)

        elif isinstance(other, Signal):
            if other.xs != self.xs:
                logger.error(
                    message := (
                        "Can only convolve signals with the same time step"
                    )
                )
                raise ValueError(message)
            longer = self if self.n > other.n else other
            shorter = other if longer == self else self
            time = pd.Index(self.xs * np.arange(longer.n - shorter.n + 1))
            conv = np.convolve(self.y, other.y, mode="valid")

            return Signal(x=time, y=conv, name=f"{self.name}*{other.name}")

        elif isinstance(other, FilterBank):
            return SignalBundle(
                signals=[self * filter for filter in other.filters],
                name=f"{self.name}*{other.name}",
            )
        else:
            raise TypeError("Cannot convolve these types")

    def __truediv__(self, other):
        if isinstance(other, AnyScalar):
            return Signal(x=self.x, y=self.y / other)
        else:
            raise TypeError("Cannot divide these types")

    def __rtruediv__(self, other):
        return self.__div__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, Signal):
            if all(self.x == other.x):

                return Signal(
                    x=self.x,
                    y=self.y + other.y,
                    name=f"{self.name}+{other.name}",
                )
            else:
                logger.error(error := "indices need to be the same")
                raise ValueError(error)
        else:
            raise TypeError('Can only \\"add\\" to a signal')

    def __sub__(self, other):
        if isinstance(other, Signal):
            if all(self.x == other.x):
                return Signal(
                    x=self.x,
                    y=self.y - other.y,
                    name=f"{self.name}-{other.name}",
                )
            else:
                logger.error(error := "indices need to be the same")
                raise ValueError(error)
        else:
            raise TypeError('Can only \\"subtract\\" a signal')

    def resample(self, new_fs: u.Quantity) -> "Signal":
        """Resample the signal to a new sampling frequency.

        Uses ``scipy.signal.resample`` (FFT-based).

        Parameters
        ----------
        new_fs : astropy.units.Quantity
            Target sampling frequency.

        Returns
        -------
        Signal
            Resampled signal.
        """
        ratio = dimensionless(new_fs / self.fs)
        new_n = int(round(self.n * ratio))
        new_y = scp_signal.resample(self.y.value, new_n) * self.y_unit
        new_x = np.linspace(self.begin, self.end, new_n)
        return Signal(
            x=new_x,
            y=new_y,
            name=f"{self.name} (resampled {new_fs})",
            x_name=self.x_name,
            y_name=self.y_name,
        )

    def window(self, window_type: str = "hann") -> "Signal":
        """Apply a window function to the signal.

        Parameters
        ----------
        window_type : str, optional
            Window name supported by ``scipy.signal.get_window``
            (e.g. ``"hann"``, ``"blackman"``, ``"hamming"``).
            Default is ``"hann"``.

        Returns
        -------
        Signal
            New signal with the window applied element-wise.
        """
        w = scp_signal.get_window(window_type, self.n)
        return Signal(
            x=self.x,
            y=self.y * w,
            name=f"{self.name} ({window_type})",
            x_name=self.x_name,
            y_name=self.y_name,
        )

    def psd(self, **kwargs) -> "PSDSignal":
        """Compute the Power Spectral Density using Welch's method.

        Parameters
        ----------
        **kwargs
            Passed to ``scipy.signal.welch``.

        Returns
        -------
        PSDSignal
            PSD of the signal.
        """
        psd_frequencies, psd_values = scipy.signal.welch(
            self.y.value, fs=dimensionless(self.x.unit / self.xs), **kwargs
        )

        psd_values = psd_values * self.y_unit**2 * self.xs.unit
        psd_frequencies = psd_frequencies / self.xs.unit

        name = f"PSD of {self.name}"
        x_name = "Frequency"
        y_name = f"PSD({self.name})"
        return PSDSignal(
            x=psd_frequencies,
            y=psd_values,
            x_name=x_name,
            y_name=y_name,
            name=name,
        )


class DatedSignal(Signal):
    """A Signal with an associated origin date.

    Parameters
    ----------
    origin : pandas.DatetimeIndex
        The reference date.
    x : astropy.units.Quantity
        Time values (must have time physical type).
    y : astropy.units.Quantity
        Signal values.
    """

    def __init__(self, origin: pd.DatetimeIndex, x: u.Quantity, y: u.Quantity):
        self.origin = origin
        unit: u.Unit = cast(u.Unit, x.unit)

        if unit.physical_type != "time":
            raise ValueError("x must be a time quantity")
        super().__init__(y=y, x=x)


class ImpulseResponse(Signal):
    """Impulse response signal, plotted as a scatter plot by default.

    Parameters
    ----------
    index : array-like
        Time axis values.
    y : astropy.units.Quantity
        Impulse response values.
    """

    def __init__(self, index: pd.Index, y: u.Quantity):
        super().__init__(y=y, x=index)

    def plot(
        self,
        x_display_unit: Optional[u.Unit] = None,
        y_display_unit: Optional[u.Unit] = None,
        **kwargs,
    ):
        super().plot(
            x_display_unit=x_display_unit,
            y_display_unit=y_display_unit,
            plot_type="scatter",
        )


def apply_property_to_signals(arr: NDArray, apply_property: Callable):
    """Apply a property to each Signal in an array, preserving shape.

    Parameters
    ----------
    arr : numpy.ndarray
        Array of Signal objects (possibly multi-dimensional).
    apply_property : str
        Name of a property attribute of Signal (e.g. ``"rms"``).

    Returns
    -------
    numpy.ndarray
        Array with the property values, same shape as ``arr``.
    """

    # Function to recursively process elements of the array
    def recurse(element):
        if isinstance(element, Signal):
            # Apply the process method and return a new Signal
            return getattr(element, apply_property)
        elif isinstance(element, np.ndarray):
            # Recursively apply the function to each element and
            # form a new array
            return np.array([recurse(elem) for elem in element])
        else:
            # Return the element unchanged if it's not a Signal
            return element

    # Apply the recurse function across all elements of the input array
    # and ensure the shape is retained by reshaping at the end
    result = array_from_list([recurse(elem) for elem in arr.flat]).reshape(
        arr.shape
    )
    return result


class PSDSignal(Signal):
    """Power Spectral Density signal, plotted with log-log axes by default."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_yscale("log")
        ax.set_xscale("log")
        return ax


class OneSignal(Signal):
    """A constant unit-amplitude signal.

    Parameters
    ----------
    index : astropy.units.Quantity
        Time axis values.
    """

    def __init__(self, index: u.Quantity):
        super().__init__(y=np.ones(len(index)), x=index)


class SineSignal(Signal):
    """A sine wave signal.

    Parameters
    ----------
    index : array-like
        Time axis values.
    amplitude : float
        Peak amplitude.
    frequency : float
        Frequency of the sine wave.
    """

    def __init__(self, index: pd.Index, amplitude: float, frequency: float):
        y = amplitude * np.sin(index * 2 * π * frequency)
        super().__init__(y=y, x=index)


class WhiteNoise(Signal):
    """Band-limited white noise signal with a specified one-sided PSD.

    Parameters
    ----------
    psd : astropy.units.Quantity
        One-sided power spectral density value.
    x : astropy.units.Quantity
        Time axis values.
    unit : astropy.units.Quantity
        Physical unit of the noise (e.g. ``u.volt``).
        Must satisfy ``unit**2 / x.unit`` being dimensionless.
    name : str, optional
        Signal name.
    **kwargs
        Passed to :class:`Signal`.
    """

    def __init__(
        self,
        psd: u.Quantity,
        x: u.Quantity,
        unit: u.quantity.Quantity,
        name: Optional[str] = None,
        **kwargs,
    ):
        xs = Signal.get_xs(x)  # type: ignore
        sd = (np.sqrt(psd) / np.sqrt(2 * xs)) / (unit)  # type: ignore

        if sd.unit.physical_type != u.dimensionless_unscaled.physical_type:
            logger.error(
                error := (
                    "psd**2/x.unit must be dimensionless, "
                    f"got {sd.unit.physical_type}"
                )
            )
            raise ValueError(error)
        else:
            sd.dimensionless = sd.to(u.dimensionless_unscaled)

        y = (
            np.random.normal(  # type: ignore
                loc=0, scale=sd.dimensionless, size=len(x)
            )
            * unit
        )

        if name is None:
            name = f"White noise with PSD {psd}"
        super().__init__(x, y, name=name, **kwargs)


class FlickerNoise(Signal):
    """Flicker (1/f) noise generated by filtering white noise through a Barnes filter.

    Parameters
    ----------
    psd : astropy.units.Quantity
        PSD level of the underlying white noise.
    x : astropy.units.Quantity
        Time axis values.
    unit : astropy.units.Quantity
        Physical unit of the noise.
    f0 : astropy.units.Quantity
        Base frequency for the Barnes filter.
    fref : astropy.units.Quantity
        Reference frequency for gain normalization.
    name : str, optional
        Signal name.
    """

    instance_counter = 0
    order = 6
    K = 10
    gain = 1

    def __init__(self, psd, x, unit, f0, fref, name=None):
        self.instance_counter += 1
        finf = 5 / get_interval(x)
        white_noise = WhiteNoise(psd, x, unit, name)
        self.filter = BarnesFilter(
            f0=f0,
            finf=finf,
            n=self.order,
            K=self.K,
            fs=white_noise.fs,
            gain=1,
            fref=fref,
            name="barnes filter",
        )

        if name is None:
            name = f"Flicker noise with PSD {psd}@{fref}"

        y = (
            scipy.signal.lfilter(
                b=self.filter.b,
                a=self.filter.a,
                x=white_noise.y,
            )
            * unit
        )
        self._psd = psd
        self.fref = fref
        self.f0 = f0
        super().__init__(y=y, x=x, name=name)


class FrequencyResponseSignal(Signal):
    """Frequency response signal, plotted with log-log axes by default."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(
        self,
        x_display_unit: Optional[u.Unit] = None,
        y_display_unit: Optional[u.Unit] = None,
        **kwargs,
    ):
        ax = FlexibleSignal.plot(
            self,
            x_display_unit=x_display_unit,
            y_display_unit=y_display_unit,
            **kwargs,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        return ax


class Filter(object):
    """Abstract base class for all filters."""

    @abstractmethod
    def apply(self, signal: Signal):
        """Apply the filter to a signal."""
        pass


class LinearFilter(Filter):
    """A discrete-time linear filter defined by numerator (b) and denominator (a) coefficients.

    Supports frequency response analysis, Bode plots, and impulse response
    computation. Use ``signal * filter`` to apply.

    Parameters
    ----------
    b : numpy.ndarray or float or int
        Numerator (feedforward) coefficients.
    a : numpy.ndarray or float or int, optional
        Denominator (feedback) coefficients. Default is 1 (FIR filter).
    name : str, optional
        Filter name.

    Attributes
    ----------
    b : numpy.ndarray
        Numerator coefficients.
    a : numpy.ndarray or int
        Denominator coefficients.
    """

    instance_counter = 0

    def generate_name(self):
        """str : Auto-generated filter name."""
        return f"Linear filter N. {self.instance_counter}"

    def __init__(
        self,
        b: Union[np.ndarray, float, int] = np.ndarray([1]),
        a: Optional[Union[np.ndarray, float, int]] = None,
        name: Optional[str] = "",
    ):

        LinearFilter.instance_counter += 1
        if not name or name == "":
            self.name = self.generate_name()
        else:
            self.name = name

        # accept scalar b values
        if isinstance(b, (float, int)):
            b = np.array([b])
        else:
            b = np.array(b)

        # accept scalar a values
        if isinstance(a, (float, int)):
            a = np.array([a])

        elif a is not None:
            a = np.array(a)
        else:
            a = 1

        self.a = a
        self.b = b

    def apply(self, signal: Signal):
        """Apply the filter to a signal.

        Parameters
        ----------
        signal : Signal
            Input signal.

        Returns
        -------
        Signal
            Filtered signal.
        """
        return signal * self

    def freqz(self, fs: u.quantity.Quantity, **kwargs):
        """Compute the frequency response using ``scipy.signal.freqz``.

        Parameters
        ----------
        fs : astropy.units.Quantity
            Sampling frequency.
        **kwargs
            Passed to ``scipy.signal.freqz``.

        Returns
        -------
        w : astropy.units.Quantity
            Frequency array.
        h : numpy.ndarray
            Complex frequency response.
        """
        w, h = scipy.signal.freqz(
            b=self.b, a=self.a, fs=fs.value, include_nyquist=True, **kwargs
        )
        return w * fs.unit, h

    def frequency_response(
        self, fs: Union[u.quantity.Quantity, float, int], **kwargs
    ):
        """Compute the magnitude frequency response.

        Parameters
        ----------
        fs : astropy.units.Quantity or float
            Sampling frequency.
        **kwargs
            Passed to :meth:`freqz`.

        Returns
        -------
        FrequencyResponseSignal
            Magnitude response.
        """
        w, h = self.freqz(fs=fs, **kwargs)
        y = np.abs(h) * u.dimensionless_unscaled

        return FrequencyResponseSignal(
            x=w,
            y=y,
            name=f"Frequency response, {self.name}, fs={fs}",
            x_name="Frequency",
            y_name="$|H(j2\\pi f)|$",
        )

    def bode(
        self, fs: Union[u.quantity.Quantity, float, int], **kwargs
    ) -> Tuple["FrequencyResponseSignal", "FrequencyResponseSignal"]:
        """Compute magnitude and phase response (Bode plot data).

        Parameters
        ----------
        fs : astropy.units.Quantity or float
            Sampling frequency.
        **kwargs
            Passed to :meth:`freqz`.

        Returns
        -------
        magnitude : FrequencyResponseSignal
            Magnitude response.
        phase : FrequencyResponseSignal
            Phase response in degrees.
        """
        w, h = self.freqz(fs=fs, **kwargs)
        magnitude = np.abs(h) * u.dimensionless_unscaled
        phase = np.angle(h, deg=True) * u.deg

        mag_signal = FrequencyResponseSignal(
            x=w,
            y=magnitude,
            name=f"Magnitude, {self.name}, fs={fs}",
            x_name="Frequency",
            y_name="$|H(j2\\pi f)|$",
        )
        phase_signal = FrequencyResponseSignal(
            x=w,
            y=phase,
            name=f"Phase, {self.name}, fs={fs}",
            x_name="Frequency",
            y_name="$\\angle H(j2\\pi f)$",
        )
        return mag_signal, phase_signal

    def __rmul__(self, other):
        if isinstance(other, Signal):
            return other.__mul__(self)
        return NotImplemented

    def impulse_response(
        self,
        pre_padding: int = 15,
        post_padding=15,
        fs: u.quantity.Quantity = 1,
        x_unit: u.quantity.Quantity = u.dimensionless_unscaled,
    ):
        """Compute the impulse response of the filter.

        Parameters
        ----------
        pre_padding : int, optional
            Number of zero samples before the impulse. Default is 15.
        post_padding : int, optional
            Number of zero samples after the impulse. Default is 15.
        fs : astropy.units.Quantity, optional
            Sampling frequency. Default is 1.
        x_unit : astropy.units.Quantity, optional
            Unit for the impulse. Default is dimensionless.

        Returns
        -------
        ImpulseResponse
            The filter's impulse response.
        """
        time = np.arange(-pre_padding, post_padding + 1) * (1 / fs)
        x = (
            np.hstack([np.zeros(pre_padding), 1, np.zeros(post_padding)])
            * x_unit
        )
        h = self * Signal(x=time, y=x)
        return ImpulseResponse(y=h.y, index=time)


class MovingAverageFilter(LinearFilter):
    """Moving average (boxcar) FIR filter.

    Parameters
    ----------
    n : int
        Number of samples to average.
    **kwargs
        Passed to :class:`LinearFilter` (e.g. ``name``).
    """

    def generate_name(self):
        return f"Moving average filter n={self.n} N.{self.instance_counter}"

    def __init__(self, n: int, **kwargs):
        self.n = n
        b = np.ones(n) / n
        super().__init__(b=b, **kwargs)


class FBMFlickerFilter(LinearFilter):
    """Fractional Brownian Motion (FBM) flicker filter.

    Generates filter coefficients that approximate a 1/f spectral shape.

    Parameters
    ----------
    order : int, optional
        Filter order (number of taps). Default is 20.
    name : str, optional
        Filter name.
    """

    @classmethod
    def fbm_filter(cls, order: int = 20, beta=-1):
        """Compute FBM filter coefficients.

        Parameters
        ----------
        order : int, optional
            Number of taps. Default is 20.
        beta : float, optional
            Spectral exponent. Default is -1 (flicker).

        Returns
        -------
        b : numpy.ndarray
            Numerator coefficients.
        a : numpy.ndarray
            Denominator coefficients (always ``[1]``).
        """
        b = np.zeros(order)
        b[0] = 1
        b_a = 1
        for k in range(1, int(order)):
            b[k] = (k - 1 - beta / 2.0) * b_a / float(k)
            b_a = b[k]
            if k % 1000 == 0:
                print(f"FBM generation:{k}")
        return b, np.asarray([1])

    def __init__(self, order: int = 20, name=None):
        self.b, self.a = self.fbm_filter(order=order)
        super().__init__(b=self.b, a=self.a, name=f"FBM order {order} filter")


class BarnesFilter(LinearFilter):
    """Barnes bilinear IIR filter for flicker noise generation.

    Parameters
    ----------
    f0 : astropy.units.Quantity
        Base frequency.
    finf : astropy.units.Quantity
        Upper limit frequency.
    n : int
        Number of filter stages.
    K : float
        Filter parameter.
    fs : astropy.units.Quantity
        Sampling frequency.
    gain : float
        Desired gain.
    fref : astropy.units.Quantity
        Reference frequency for gain normalization.
    **kwargs
        Passed to :class:`LinearFilter` (e.g. ``name``).
    """

    def __init__(
        self,
        f0: u.Quantity,
        finf: u.Quantity,
        n: AnyNumber,
        K: AnyNumber,
        fs: u.Quantity,
        gain: AnyNumber,
        fref: u.Quantity,
        **kwargs,
    ):

        b, a = barnes_bilinear(
            f0=f0,
            finf=finf,
            n=n,
            K=K,
            fs=fs,
            gain=gain,
            fref=fref,
        )
        super().__init__(b=b, a=a, **kwargs)


class AllanFilter(LinearFilter):
    """Allan variance filter.

    Computes the difference between successive non-overlapping averages
    of length ``n``, scaled by ``1/(2n)``.

    Parameters
    ----------
    n : int
        Averaging length (in samples).
    **kwargs
        Passed to :class:`LinearFilter` (e.g. ``name``).
    """

    def __init__(self, n: int, **kwargs):
        self.n = n
        b = np.concatenate((np.ones(n), -np.ones(n))) / (2 * self.n)
        if "name" not in kwargs:
            kwargs["name"] = f"Allan filter (n={self.n})"
        super().__init__(b=b, **kwargs)


class PVarFilter(LinearFilter):
    """Parabolic variance (PVar) filter.

    Parameters
    ----------
    p : int
        Half-width of the filter kernel.
    """

    def __init__(self, p: int):
        self.p = p
        b = np.array([k for k in range(-p, p + 1, 1)])
        super().__init__(b=b)


class SimpleIntegrator(LinearFilter):
    """Simple discrete-time integrator: ``H(z) = 1 / (1 + z^-1)``."""

    def __init__(self):
        super().__init__(b=np.array([1]), a=np.array([1, 1]))


class FilterBank:
    """A collection of filters applied collectively to a signal.

    ``signal * filterbank`` returns a :class:`SignalBundle` with one
    signal per filter.

    Parameters
    ----------
    filters : list or numpy.ndarray
        Sequence of :class:`Filter` instances.
    name : str, optional
        Name for the filter bank.
    """

    instance_counter = 0

    @classmethod
    def generate_name(cls):
        """str : Auto-generated filter bank name."""
        return f"{cls.__name__}_{cls.instance_counter}"

    def __init__(
        self,
        filters: Union[np.ndarray, List[Filter]],
        name: Optional[str] = None,
    ):
        if not name:
            self.name = self.generate_name()
        else:
            self.name = name

        if isinstance(filters, list):
            filters = np.array(filters)
        self.filters = filters

    def __getitem__(self, item):
        return self.filters[item]

    def __mul__(self, other):
        if isinstance(other, AnyScalar):
            return FilterBank(
                filters=[filter * other for filter in self.filters],
                name=self.name,
            )
        elif isinstance(other, Filter):
            return FilterBank(
                filters=[filter * other for filter in self.filters],
                name=self.name,
            )
        elif isinstance(other, Signal):
            return SignalBundle(
                signals=[other * afilter for afilter in self.filters]
            )
        else:
            raise TypeError('Can only \\"multiply\\" by a number or a filter')

    def __rmul__(self, other):
        return self.__mul__(other)


class SignalBundle:
    """A container for multiple signals with collective plotting and analysis.

    Supports ``bundle * filter``, ``bundle * filterbank``, iteration,
    indexing, and both overlay and subplot-grid plotting.

    Parameters
    ----------
    signals : list or numpy.ndarray
        Sequence of :class:`Signal` instances.
    name : str, optional
        Name for the bundle.
    """

    instance_counter = 0
    figsize = (12, 10)

    def __init__(
        self, signals: Union[List, np.ndarray], name: Optional[str] = None
    ):
        self.instance_counter += 1

        if isinstance(signals, list):
            signals = np.array(signals)
        self.signals = signals

        if not name:
            self.name = f"Signal bundle {self.instance_counter}"
        else:
            self.name = name

        self.n = len(self.signals)

    def __repr__(self):
        signals_repr = [f"{signal!r}" for signal in self.signals]
        return (
            f"Signal bundle {self.name} with {self.n} signals: {signals_repr}"
        )

    def __len__(self):
        return len(self.signals)

    def __iter__(self):
        return iter(self.signals)

    def __str__(self):
        return f"Signal bundle {self.name} with {self.n} signals"

    def __getitem__(self, item):
        return self.signals[item]

    def __mul__(self, other):
        if isinstance(other, LinearFilter):
            filtered_signal_array = np.array(
                [signal * other for signal in self.signals]
            )
            return SignalBundle(signals=filtered_signal_array)

        elif isinstance(other, FilterBank):
            return SignalBundle(
                [signal * other for signal in self.signals],
                self.name,
            )

    def __matmul__(self, other):
        """Convolves a signal bundle with either a number of a filter bank
        using the same rules as matrix multiplication.

        :param other: _description_
        :type other: _type_
        :raises TypeError: _description_
        :return: _description_
        :rtype: _type_
        """
        if isinstance(other, numbers.Number):
            return SignalBundle(
                signals=self.signals * other,
                name=self.name,
            )
        elif isinstance(other, FilterBank):
            return SignalBundle(
                signals=self.signals @ other.filters,
                name=f"Signal bundle{self.instance_counter}",
            )
        else:
            raise TypeError('Can only \\"multiply\\" by a number or a filter')

    @property
    def signal_labels(self):
        return [signal.name for signal in self.signals]

    def plot(
        self,
        all_in_one=True,
        signal_labels: List[str] = [],
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        suptitle: Optional[str] = None,
        x_display_unit: Optional[u.Unit] = None,
        y_display_unit: Optional[u.Unit] = None,
        **kwargs,
    ):
        if suptitle is None:
            suptitle = self.name

        if signal_labels == []:
            signal_labels = self.signal_labels

        keyword_arguments = kwargs.copy()
        if all_in_one:
            fig, ax = plt.subplots()
            for idx, signal in enumerate(self.signals):
                keyword_arguments["label"] = signal.name

                scaled_x, scaled_x_label = signal.x_scale_display(
                    x_display_unit, x_label=x_label
                )
                scaled_y, scaled_y_label = signal.y_scale_display(
                    y_display_unit, y_label=y_label
                )

                plot_type = kwargs.pop("plot_type", "line")
                signal.plot_bare(
                    ax=ax,
                    # **keyword_arguments,
                    scaled_x=scaled_x,
                    scaled_y=scaled_y,
                    x_display_unit=x_display_unit,
                    y_display_unit=y_display_unit,
                    label=signal_labels[idx],
                    plot_type=plot_type,
                )
                ax.set_xlabel(scaled_x_label)
                ax.set_ylabel(scaled_y_label)

            ax.set_title(suptitle)
            # Set custom labels if provided in kwargs
            if "x_label" in kwargs:
                ax.set_xlabel(kwargs.pop("x_label"))

            if "y_label" in kwargs:
                ax.set_ylabel(kwargs.pop("y_label"))

            ax.grid(which="both", axis="both")
            ax.legend()

        else:
            n = len(self.signals)
            ratio = 4 / 3
            ncols = int(np.ceil(np.sqrt(ratio * n)))
            nrows = int(np.ceil(3 * np.floor(np.sqrt(ratio * n)) / 4))

            if ncols * nrows < n or ncols * nrows > n + ncols:
                raise ValueError("Not enough subplots to plot all signals")

            fig, axs = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=self.figsize
            )
            fig.subplots_adjust(wspace=0.2, hspace=0.3)

            flat_ax = axs.flatten()

            if ncols * nrows == n + 1:
                fig.delaxes(flat_ax[-1])

            for index, signal in enumerate(self.signals):
                keyword_arguments["label"] = signal.name
                ax = flat_ax[index]

                scaled_x, scaled_x_label = signal.x_scale_display(
                    x_display_unit
                )
                scaled_y, scaled_y_label = signal.y_scale_display(
                    y_display_unit
                )

                (self.signals[index]).plot_bare(
                    ax=ax,
                    scaled_x=scaled_x,
                    scaled_y=scaled_y,
                    x_display_unit=x_display_unit,
                    y_display_unit=y_display_unit,
                    **keyword_arguments,
                )
                ax.set_xlabel(signal.x_name)
                ax.set_ylabel(signal.y_name)
                ax.legend()
                ax.grid(which="both", axis="both")
                ax.set_title(signal_labels[index])

            fig.suptitle(suptitle)
        return fig, ax


STOCK_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
USD = u.def_unit("USD")
STOCK_UNITS = {
    "Open": USD,
    "High": USD,
    "Low": USD,
    "Close": USD,
    "Volume": u.dimensionless_unscaled,
}


class StockFlexibleSignal(FlexibleSignal):
    """A FlexibleSignal fetched from Yahoo Finance.

    Data points correspond to trading days only (non-uniform spacing).

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    column : str, optional
        Data column: ``"Open"``, ``"High"``, ``"Low"``, ``"Close"``,
        or ``"Volume"``. Default is ``"Close"``.
    period : str, optional
        Data period (e.g. ``"1y"``, ``"6mo"``). Default is ``"1y"``.
    start : str, optional
        Start date (e.g. ``"2024-01-01"``). Overrides ``period``.
    end : str, optional
        End date. Used with ``start``.
    """

    def __init__(
        self,
        ticker: str,
        column: str = "Close",
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        import yfinance as yf

        if column not in STOCK_COLUMNS:
            raise ValueError(
                f"column must be one of {STOCK_COLUMNS}, got '{column}'"
            )

        self.ticker = ticker
        self.column = column
        t = yf.Ticker(ticker)

        if start is not None:
            history = t.history(start=start, end=end)
        else:
            history = t.history(period=period)

        if history.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")

        self._history = history

        origin = history.index[0]
        x_days = (
            (history.index - origin) / pd.Timedelta(days=1)
        ).values * u.day
        y_values = history[column].values * STOCK_UNITS.get(
            column, u.dimensionless_unscaled
        )

        super().__init__(
            x=x_days,
            y=y_values,
            name=f"{ticker} {column}",
            x_name=f"Days since {origin.strftime('%Y-%m-%d')}",
            y_name=column,
        )
        self.origin = origin

    @property
    def history(self) -> pd.DataFrame:
        """The full Yahoo Finance history DataFrame."""
        return self._history


class StockSignal(Signal):
    """A uniformly-sampled (1-day) stock signal, interpolated from trading days.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    column : str, optional
        Data column. Default is ``"Close"``.
    period : str, optional
        Data period (e.g. ``"1y"``, ``"6mo"``). Default is ``"1y"``.
    start : str, optional
        Start date (e.g. ``"2024-01-01"``). Overrides ``period``.
    end : str, optional
        End date. Used with ``start``.
    kind : str, optional
        Interpolation method (e.g. ``"linear"``, ``"cubic"``).
        Default is ``"linear"``.
    """

    def __init__(
        self,
        ticker: str,
        column: str = "Close",
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        kind: str = "linear",
    ):
        stock_fs = StockFlexibleSignal(
            ticker=ticker,
            column=column,
            period=period,
            start=start,
            end=end,
        )
        self.origin = stock_fs.origin
        self.ticker = ticker
        self.column = column
        self._history = stock_fs.history

        signal = stock_fs.to_signal(fs=1 / u.day, kind=kind)
        super().__init__(
            x=signal.x,
            y=signal.y,
            name=signal.name,
            x_name=stock_fs.x_name,
            y_name=stock_fs.y_name,
        )


class StockReturnSignal(Signal):
    """A uniformly-sampled (1-day) daily return signal from Yahoo Finance.

    Computes relative daily changes ``r(t) = (p(t) - p(t-1)) / p(t-1)``
    from interpolated closing prices. The resulting signal is dimensionless.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    column : str, optional
        Data column. Default is ``"Close"``.
    period : str, optional
        Data period (e.g. ``"1y"``, ``"6mo"``). Default is ``"1y"``.
    start : str, optional
        Start date (e.g. ``"2024-01-01"``). Overrides ``period``.
    end : str, optional
        End date. Used with ``start``.
    kind : str, optional
        Interpolation method. Default is ``"linear"``.
    """

    def __init__(
        self,
        ticker: str,
        column: str = "Close",
        period: str = "1y",
        start: Optional[str] = None,
        end: Optional[str] = None,
        kind: str = "linear",
    ):
        stock = StockSignal(
            ticker=ticker,
            column=column,
            period=period,
            start=start,
            end=end,
            kind=kind,
        )
        self.origin = stock.origin
        self.ticker = ticker
        self.column = column
        self._history = stock._history

        returns = np.diff(stock.y.value) / stock.y.value[:-1]
        super().__init__(
            x=stock.x[1:],
            y=returns * u.dimensionless_unscaled,
            name=f"{ticker} daily return",
            x_name=stock.x_name,
            y_name="Daily return",
        )
