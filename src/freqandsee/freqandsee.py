from pathlib import Path
import pandas as pd
from typing import Union, Optional, List, cast, Callable, Tuple

from scipy import signal as scp_signal  # type: ignore
import numpy as np
from numpy.typing import NDArray

import numbers
import scipy
import astropy.units as u  # type: ignore
import matplotlib.pyplot as plt
from math import pi as π
from numpy.polynomial.polynomial import polyval
from typing import TypeAlias, Callable
from abc import abstractmethod

from freqandsee.project_logger import logger

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)
AnyNumber: TypeAlias = numbers.Number | float | int
AnyScalar = (numbers.Number, float, int, u.Unit, u.Quantity)

RELATIVE_EQUALITY_THRESHOLD = 1e-11


physical_types_units = {"frequency": u.hertz}


def bilinear_transform(pole, fs):
    """
    Perform a bilinear transformation on the given pole value.

    Parameters:
        pole: The pole value to transform.
        fs: The sampling frequency.

    Returns:
        The transformed pole value.
    """
    return -(-pole - 2 * fs) / (-pole + 2 * fs)


def hollos(n):
    """
    Compute the transfer function coefficients using the Hollos method.

    Parameters:
        n: Number of poles/zeros to generate.

    Returns:
        A tuple (b, a) where b and a are the numerator and denominator
        coefficients.
    """
    bk = [np.cos(2 * k * np.pi / float(2 * n + 1)) for k in range(1, n + 1)]
    bk = np.asarray(bk)

    zzeros = bk
    zpoles = -bk

    b, a = scp_signal.zpk2tf(zzeros, zpoles, 1)
    return b, a


def dimensionless(x: u.Quantity):
    """
    Convert the given quantity to a dimensionless value.

    Parameters:
        x: An astropy Quantity expected to have a dimensionless unit.

    Returns:
        The numeric value of x after converting to a dimensionless unit.

    Raises:
        ValueError: If x does not have a dimensionless unit.
    """
    unit = x.unit if x.unit is not None else u.dimensionless_unscaled
    if not unit.physical_type == "dimensionless":
        raise ValueError(
            "The unit of the input must be dimensionless, "
            "got {}".format(unit)
        )
    return x.to(u.dimensionless_unscaled).value


def barnes_bilinear(f0, finf, n, K, fs, gain, fref):
    """
    Compute the poles and zeros of a Barnes bilinear filter.

    Parameters:
        f0: Base frequency.
        finf: Upper limit frequency.
        n: Number of filter stages.
        K: Filter parameter.
        fs: Sampling frequency.
        gain: Desired gain.
        fref: Reference frequency.

    Returns:
        A tuple (b, a) of filter coefficients.
    """
    xs = 1 / fs
    alpha = np.float128(-1)
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
    """
    Convert a list of astropy Quantities to a numpy array while preserving its unit.

    Parameters:
        homogenous_list: A list of astropy Quantity objects that share the same unit.

    Returns:
        A numpy ndarray of quantities with the common unit.

    Raises:
        ValueError: If the elements have differing units.
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
    """
    Determine the interval between successive values in a sequence.

    Parameters:
        index: A sequence or index (possibly an astropy Quantity) where the interval is computed.

    Returns:
        The interval between elements if constant, otherwise -1.
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
    """
    Represents a flexible signal composed of an x and y component, each of which can carry units and associated metadata.
    It provides functionality for data analysis (e.g., computing statistical metrics like RMS, variance, standard deviation)
    and for visualizing the signal (with support for unit scaling and both linear and logarithmic plotting).
    Attributes:
        x: The x-data of the signal. This is expected to be a numpy array or a quantity with units.
        y: The y-data of the signal. This is expected to be a numpy array or a quantity with units.
        name: A descriptive name for the signal. If not provided, it is automatically generated.
        x_name: The display name for the x-axis. If not provided, it is automatically generated.
        y_name: The display name for the y-axis. If not provided, it is automatically generated.
        instance_counter (class attribute): A counter used to generate default names for signal components.
        to_string_options (class attribute): A dictionary of options used when converting unit objects to strings.
    Methods:
        __init__(self, x, y, name=None, x_name=None, y_name=None):
            Initializes the signal with x and y data, sets the names, and determines the corresponding units.
        to_df(self) -> pd.DataFrame:
            Returns a pandas DataFrame containing the signal data in a unitless format.
        to_csv(self, path: Union[Path, str], add_comment=True):
            Exports the signal data to a CSV file. If add_comment is True, a header with signal names and units
            is added as a comment.
        x_unit_latexstring(self):
            Returns the LaTeX-formatted representation of the x unit.
        y_unit_latexstring(self):
            Returns the LaTeX-formatted representation of the y unit.
        x_unit_string(self):
            Returns the unicode string representation of the x unit.
        y_unit_string(self):
            Returns the unicode string representation of the y unit.
        rms(self):
            Calculates and returns the root mean square of the y-data.
        var(self):
            Returns the variance of the y-data, computed as the mean of the squared y values.
        rmsfunc(self):
            Computes and returns the root mean square (RMS) of the y-data.
        std(self):
            Calculates the standard deviation of the y-data.
        mean(self):
            Computes the mean value of the y-data.
        n(self):
            Returns the number of data points in the x array (which should equal the length of y).
        fs(self):
            A property that represents the sampling frequency of the signal (if defined).
        unitless_values(self):
            Returns the y-data stripped of any associated units.
        get_units(cls, x: Union[np.ndarray, u.Quantity], y: Union[np.ndarray, u.Quantity]):
            A class method that inspects the first element of x and y to determine and return their respective units.
            It also validates that x and y have the same length.
        x_scale_display(self, x_display_unit: u.Unit):
            Scales the x-data for display according to the provided unit and returns a tuple consisting of the
            scaled data and a formatted label including the unit in a readable format.
        y_scale_display(self, y_display_unit: u.Unit):
            Scales the y-data for display according to the provided unit and returns a tuple consisting of the
            scaled data and a formatted label including the unit in a readable format.
        loglogplot(self, **kwargs):
            Generates a plot of the signal with both the x and y axes set to logarithmic scales. Returns the axis object.
        plot(self, x_display_unit: Optional[u.Unit] = None, y_display_unit: Optional[u.Unit] = None, **kwargs):
            Plots the signal with optional unit scaling for the x and y axes. Supports customization options such as
            plot type (line or scatter), title, and use of an existing Matplotlib axis. Returns the axis object.
    Properties:
        name (getter/setter):
            A property to get or set the signal's name.

    """

    instance_counter = 0
    to_string_options = {"format": "latex", "fraction": False}

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
        self.instance_counter += 1

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
        """
        Return a DataFrame with the signal data (unitless both in x and y).

        Returns:
            pd.DataFrame: A DataFrame with two columns representing the x
            and y data.
        """
        return pd.DataFrame(
            {self.x_name: self.x.value, self.y_name: self.y.value}
        )

    @classmethod
    def from_csv(cls, path: Path) -> "FlexibleSignal":
        """Read the signal data from a CSV file with an optional header
        containing units.

        This method reads the signal data from a CSV file and converts it into
        a pandas DataFrame. If the file contains a header line with signal
        names and units, it will be parsed to extract this information.

        Parameters:
            path (Path): The filesystem path to the CSV file to read.
                This should be a pathlib.Path object.

        Returns:
            None

        Side Effects:
            Reads data from the specified file path.
        """
        with open(path, "r") as csv_file:
            header = csv_file.readline()
            header_dict = eval(header[2:])

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
        """
        Exports the object's data to a CSV file.

        This method converts the object's data into a pandas DataFrame using the
        to_df() method and writes it to the specified file path in CSV format. If
        requested, it also adds a comment header to the file that contains metadata
        for the x and y axes, including their names and units.

        Parameters:
            path (Union[Path, str]): The file path where the CSV will be saved. This can
                be provided as either a string or a Path object.
            add_comment (bool, optional): If True, a comment line containing metadata about
                the x and y axes (such as their names and units) will be added as the first
                line in the file. Defaults to True.

        Notes:
            - If the provided path is a string, it is converted to a Path object.
            - The comment header is written to the file as a line starting with '#'.
            - The DataFrame's CSV representation (with headers and no index) is appended after
              the comment header.

        Returns:
            None
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
        return self.x.unit.to_string(**self.to_string_options)

    @property
    def y_unit_latexstring(self):
        return self.y.unit.to_string(**self.to_string_options)

    @property
    def x_unit_string(self):
        return self.x.unit.to_string(format="unicode")

    @property
    def y_unit_string(self):
        return self.y.unit.to_string(format="unicode")

    @property
    def rms(self):
        return np.sqrt(np.mean(self.y**2))

    @property
    def var(self):
        return np.mean(self.y**2)

    def rmsfunc(self):
        return np.sqrt(np.mean(self.y**2))

    @property
    def std(self):
        return np.std(self.y)

    @property
    def mean(self):
        return np.mean(self.y)

    @property
    def n(self):
        return len(self.x)

    @property
    def fs(self):
        return self._fs

    @property
    def unitless_values(self):
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

    def x_scale_display(
        self, x_display_unit: u.Unit, x_label: Optional[str] = None
    ):
        if x_label is None:
            x_label = self.x_name
        scaled_x = dimensionless(self.x / x_display_unit)

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

    def loglogplot(self, **kwargs):
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
        """ """

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
        """ """

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
    """A Signal represents a time-series with strict constraints on its x-axis
    values.
    The x-axis must consist of monotonically increasing values with a constant
    increment, which ensures that operations such as the Discrete Fourier
    Transform (DFT) and the Power Spectral Density (PSD) are well-defined.

    Parameters
    ----------
    x : astropy.units.Quantity
        An array-like object containing the time values (with proper units).
        The values must be monotonically increasing and uniformly spaced.
    y : astropy.units.Quantity
        An array-like object containing the corresponding data (signal values)
        with appropriate units.
    name : Optional[str], default=None
        An optional name for the Signal. If not provided, a name is
        automatically generated.
    x_name : Optional[str], default=None
        An optional label for the x-axis. Defaults to "x({signal name})" if not
        provided.
    y_name : Optional[str], default=None
        An optional label for the y-axis. Defaults to "y({signal name})" if not
        provided.

    Attributes
    ----------
    begin : numeric
        The initial value from the x-axis.
    end : numeric
        The last value from the x-axis.
    _length : numeric
        The difference between the last and first x-axis values.
    x_unit : astropy.units.Unit
        The unit associated with the x values, as determined by get_units.
    y_unit : astropy.units.Unit
        The unit associated with the y values, as determined by get_units.
    xs : numeric
        The constant interval between successive x values.
    _fs : numeric
        The sampling frequency calculated as the reciprocal of xs.
    x : astropy.units.Quantity
        The complete time series data with units.
    y : astropy.units.Quantity
        The complete signal values corresponding to the time series.
    _instance_counter : int
        A class level counter used to generate default names for signal
        instances.

    Methods
    -------
    from_timed_dataframe(df: pandas.DataFrame, y_unit: astropy.units.Unit)
        Creates a Signal instance from a pandas DataFrame that has a
        DatetimeIndex.
    __getitem__(index)
        Enables indexing/slicing and returns a new Signal instance
        corresponding to the sliced data.
    __repr__()
        Returns a string representation of the Signal, including its name,
        units, number of data points, sampling frequency, and RMS value.
    fft()
        Computes the Fast Fourier Transform (FFT) of the signal and returns a
        new Signal instance representing the frequency domain version of the
        original signal.
    __mul__(other)
        Supports multiplication of the Signal by a scalar, a linear filter, or
        by performing convolution with another Signal instance (provided they
        share the same time step).
    __truediv__(other)
        Supports division of the Signal by a scalar.
    __add__(other)
        Adds two Signal instances element-wise (provided their x indices are
        identical).
    psd(**kwargs) -> PSDSignal
        Computes the Power Spectral Density (PSD) of the signal using Welch's
        method and returns a PSDSignal instance.
    get_xs(index)
        A class method that verifies and returns the uniform interval between
        x-axis values for a given index.

    Notes
    -----
    The Signal class enforces uniform time steps and monotonically increasing x
    values, which is crucial for correct signal processing operations such as
    FFT, convolution, and PSD estimation.
    """

    instance_counter = 0

    @staticmethod
    def from_timed_dataframe(df: pd.DataFrame, y_unit: u.Unit):
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
        self.xs = self.get_xs(self.x)
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
                header_dict = eval(header[2:])
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
            except:
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
        # done from scratch to handle astropy units gracefully,
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
        return applied_function(self.y.value)

    def fft(self):
        fft_values = scipy.fft.fftshift(scipy.fft.fft(self.series.values))
        frequencies = self.fftfreq * self.unit**2 * self.xs.unit
        return Signal(
            pd.Series(
                fft_values * self.unit * self.xs.unit,
                index=pd.Index(frequencies),
            )
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
            return Signal(self.y / other, x=self.x)
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

    def psd(self, **kwargs) -> "PSDSignal":
        """Uses scipy.signal.welch to create a PSDSignal instance
        with the PSD of the signal and the frequencies the PSD is calculated
        at, using the signal's sampling frequency.

        :return: PSD of the signal
        :rtype: PSDSignal
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
    def __init__(self, origin: pd.DatetimeIndex, x: u.Quantity, y: u.Quantity):
        self.origin = origin
        unit: u.Unit = cast(u.Unit, x.unit)

        if unit.physical_type != "time":
            raise ValueError("x must be a time quantity")
        super().__init__(y=y, x=x)


class ImpulseResponse(Signal):
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
    """Creates a numpy array with the same shape as the input array,
    but with each element being a Signal with the specified property
    applied.

    :param arr: input
    :type arr: np.ndarray()

    :param apply_function: must be a property attribute of Signal
    :type apply_function: Callable

    :return: array with each element being a Signal with the specified
        property applied

    :rtype: np.ndarray

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, **kwargs):
        ax = super().plot(**kwargs)
        ax.set_yscale("log")
        ax.set_xscale("log")
        return ax


class OneSignal(Signal):
    def __init__(self, index: u.Quantity):
        super().__init__(y=np.ones(len(index)), x=index)


class SineSignal(Signal):
    def __init__(self, index: pd.Index, amplitude: float, frequency: float):
        y = amplitude * np.sin(index * 2 * π * frequency)
        super().__init__(y=y, x=index)


class WhiteNoise(Signal):
    def __init__(
        self,
        psd: u.Quantity,
        x: u.Quantity,
        unit: u.quantity.Quantity,
        name: Optional[str] = None,
        **kwargs,
    ):
        """_summary_

        :param psd: Value of the PSD of the noise, one-sided.
        :type psd: u.Quantity
        :param x: index timepoints
        :type x: u.Quantity
        :param unit: unit of the noise, unit**2/<unit of x> must have same
        dimensions.
        :type unit: Optional[u.quantity.Quantity]
        :param name: suiex, defaults to None
        :type name: Optional[str], optional
        :raises ValueError: _description_
        """
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
    @abstractmethod
    def apply(self, signal: Signal):
        pass


class LinearFilter(Filter):
    instance_counter = 0

    def generate_name(self):
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
        return self * signal

    def freqz(self, fs: u.quantity.Quantity, **kwargs):
        w, h = scipy.signal.freqz(
            b=self.b, a=self.a, fs=fs.value, include_nyquist=True, **kwargs
        )
        return w * fs.unit, h

    def frequency_response(
        self, fs: Union[u.quantity.Quantity, float, int], **kwargs
    ):
        w, h = self.freqz(fs=fs, **kwargs)
        y = np.abs(h) * u.dimensionless_unscaled

        return FrequencyResponseSignal(
            x=w,
            y=y,
            name=f"Frequency response, {self.name}, fs={fs}",
            x_name="Frequency",
            y_name="$|H(j2\\pi f)|$",
        )

    def impulse_response(
        self,
        pre_padding: int = 15,
        post_padding=15,
        fs: u.quantity.Quantity = 1,
        x_unit: u.quantity.Quantity = u.dimensionless_unscaled,
    ):
        """Impulse response linear filter.

        Keyword Arguments:
            pre_padding -- zero-padding beffore the 1/n values (default: {15})
            post_padding -- zero-padding after the strictly positive values
            (default: {15})
            fs -- sampling frequency (default: {1})
            x_unit -- unit (default: {u.dimensionless_unscaled})

        Returns:
            _description_
        """
        time = np.arange(-pre_padding, post_padding + 1) * (1 / fs)
        x = (
            np.hstack([np.zeros(pre_padding), 1, np.zeros(post_padding)])
            * x_unit
        )
        h = self * Signal(x=time, y=x)
        return ImpulseResponse(y=h.y, index=time)


class MovingAverageFilter(LinearFilter):
    def generate_name(self):
        return f"Moving average filter n={self.n} N.{self.instance_counter}"

    def __init__(self, n: int, **kwargs):
        self.n = n
        b = np.ones(n) / n
        super().__init__(b=b, **kwargs)


class FBMFlickerFilter(LinearFilter):
    """Fractional brownian motion filter.

    Arguments:
        LinearFilter -- _description_

    Returns:
        _description_
    """

    @classmethod
    def fbm_filter(cls, order: int = 20, beta=-1):
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
    def __init__(self, n: int, **kwargs):
        self.n = n
        b = np.concatenate((np.ones(n), -np.ones(n))) / (2 * self.n)
        if "name" not in kwargs:
            kwargs["name"] = f"Allan filter (n={self.n})"
        super().__init__(b=b, **kwargs)


class PVarFilter(LinearFilter):
    def __init__(self, p: int):
        self.p = p
        b = np.array([k for k in range(-p, p + 1, 1)])
        super().__init__(b=b)


class SimpleIntegrator(LinearFilter):
    def __init__(self):
        super().__init__(b=np.array([1]), a=np.array([1, 1]))


class FilterBank:
    instance_counter = 0

    @classmethod
    def generate_name(cls):
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
    """A container class for an array of signals."""

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
                signal_array=self.signal_array * other,
                name=self.name,
            )
        elif isinstance(other, FilterBank):
            return SignalBundle(
                self.signal_array @ other.filter_array,
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
