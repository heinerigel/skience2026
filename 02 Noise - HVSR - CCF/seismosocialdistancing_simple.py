"""
seismosocialdistancing_simple.py
---------------------------------
Accepts a ``dataframe_RMS`` dict and plots seismic noise.

Expected input
--------------
dataframe_RMS = {
    "BE.UCC..HHZ": pd.DataFrame({"4.0-14.0": [...]}, index=pd.DatetimeIndex([...])),
    ...
}

Quick-start
-----------
>>> from seismosocialdistancing_simple import SeismoNoise
>>> sn = SeismoNoise(dataframe_RMS)
>>> sn.plot()                   # timeseries
>>> sn.plot(type="clockmaps")
>>> sn.plot(type="gridmaps")
>>> sn.plot(type="clockplots")
>>> sn.plot(type="dailyplots")
"""

import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def localize_tz_and_reindex(series, freq="30Min", time_zone="Europe/Brussels"):
    """UTC Series → local time zone, resampled."""
    return (
        series.copy()
        .tz_localize("UTC")
        .dropna()
        .tz_convert(time_zone)
        .tz_localize(None)
        .resample(freq)
        .mean()
    )


def pivot_for_hourmap(data, columns="angles"):
    """Pivot a single-column DataFrame into (day-number × hour) grid."""
    band = data.columns[0]
    data = data.copy()
    data["day"] = [d.year * 365 + d.dayofyear for d in data.index]
    data["time"] = [d.hour + d.minute / 60.0 for d in data.index]
    data = data.pivot(index="day", columns="time", values=band)
    data.index -= data.index[0]
    data.index = data.index.astype(float)
    if columns == "angles":
        data.columns = 2 * np.pi * data.columns / 24.0
    return data


def _multiindex_series(series):
    """Return Series with (weekday-name, fractional-hour) MultiIndex."""
    return pd.DataFrame(
        {"v": series.values},
        index=pd.MultiIndex.from_arrays(
            [series.index.day_name(), series.index.hour + series.index.minute / 60.0]
        ),
    )["v"]


def stack_wday_time(df, scale):
    """Median noise table: hours (rows) x weekday (columns), scaled."""
    result = df.groupby(level=(0, 1)).median().unstack(level=-1).T
    # Drop top column level only when it's a MultiIndex (i.e. came from a DataFrame)
    if isinstance(result.columns, pd.MultiIndex):
        result = result.droplevel(0, axis=1)
    # Reorder to weekdays; keep whatever columns exist
    available = [d for d in days if d in result.columns]
    if available:
        result = result[available]
    return result * scale


def radial_hours(N):
    hours = np.deg2rad(np.linspace(0, 360, N - 1, endpoint=False))
    return np.append(hours, hours[0])


def _clock_axes_style(ax, unit="nm"):
    ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
    ax.set_xticklabels(["%i h" % i for i in range(24)], fontsize=8)
    ax.set_yticklabels(["%.2g %s" % (i, unit) for i in ax.get_yticks()], fontsize=7)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_rlabel_position(0)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2.0)
    plt.xlabel("Hour (local time)", fontsize=10)
    plt.grid(True)


# ---------------------------------------------------------------------------
# Plot functions  (each takes a plain pd.Series with DatetimeIndex)
# ---------------------------------------------------------------------------

def plot_timeseries(series, label="", band="", scale=1e9, unit="nm", save=None, show=True):
    """Time-series with business-day shading and daytime median overlay."""
    fig = plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label=label)

    rs = series.between_time("6:00", "16:00").resample("1D").median().shift(12, "h")
    plt.plot(rs.index, rs, label=r"$\overline{%s}$ (6-16 h)" % label)

    for dbi in pd.bdate_range(series.index.min(), series.index.max()):
        plt.axvspan(dbi, dbi + datetime.timedelta(days=1),
                    facecolor="lightgreen", edgecolor="none", alpha=0.2, zorder=-10)

    ticks = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x * scale))
    plt.gca().yaxis.set_major_formatter(ticks)
    plt.ylim(0, np.nanpercentile(series, 95) * 1.5)
    plt.ylabel("Displacement (%s)" % unit)
    plt.title("Seismic Noise %s - [%s] Hz" % (label, band))
    plt.xlim(series.index.min(), series.index.max())
    plt.grid(True, zorder=-1)
    plt.gca().set_axisbelow(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.autofmt_xdate()

    if save:
        fig.savefig(save, bbox_inches="tight", facecolor="w")
    if show:
        plt.show()
    return fig


def plot_clockmap(series, label="", band="", scale=1e9, unit="nm", save=None, show=True):
    """Polar heat-map: day-number (radial) x hour-of-day (angular)."""
    origin_text = series.index[0].strftime("%Y-%m-%d")
    data = (series * scale).to_frame()
    vmin, vmax = data.iloc[:, 0].quantile(0.01), data.iloc[:, 0].quantile(0.95)
    data = pivot_for_hourmap(data)

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_xticks(np.linspace(0, np.pi * 2 * 23 / 24, 24))
    ax.set_xticklabels(["%d h" % h for h in range(24)])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    X = np.append(data.columns, 2 * np.pi)
    Y = np.append(data.index, data.index[-1] + 1)
    plt.pcolormesh(X, Y, data, vmax=vmax, vmin=vmin, rasterized=True, antialiased=True)
    cb = plt.colorbar(orientation="horizontal", shrink=0.8)
    cb.ax.set_xlabel("Displacement (%s)" % unit)
    ax.set_rorigin(max(Y) / -4)
    ax.text(np.pi, max(Y) / -4, origin_text, ha="center", va="center")
    ax.set_rmax(max(Y))
    ax.grid(color="w")
    plt.suptitle("Seismic Noise %s - [%s] Hz" % (label, band), fontsize=14)

    if save:
        fig.savefig(save, bbox_inches="tight", facecolor="w")
    if show:
        plt.show()
    return fig


def plot_gridmap(series, label="", band="", scale=1e9, unit="nm", save=None, show=True):
    """Cartesian heat-map: date (x) x hour-of-day (y)."""
    origin_text = series.index[0].strftime("%Y-%m-%d")
    data = (series * scale).to_frame()
    vmin, vmax = data.iloc[:, 0].quantile(0.01), data.iloc[:, 0].quantile(0.95)
    data = pivot_for_hourmap(data, columns="hours")

    fig, ax = plt.subplots(figsize=(16, 5))
    X = pd.date_range(origin_text, periods=len(data) + 1).to_pydatetime()
    Y = np.append(data.columns, 24)
    plt.pcolormesh(X, Y, data.T, vmax=vmax, vmin=vmin, rasterized=True, antialiased=True)
    plt.colorbar(shrink=0.7, pad=0.01).set_label("Displacement (%s)" % unit)
    ax.set_xticks(pd.date_range(X[0], X[-1], freq="W-MON").to_pydatetime())
    ax.set_yticks(np.arange(25))
    ax.set_yticklabels(["%d h" % h for h in range(25)])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    plt.grid(True, which="major", c="k")
    ax.set_title("Seismic Noise %s - [%s] Hz" % (label, band))
    fig.autofmt_xdate()
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12)

    if save:
        fig.savefig(save, bbox_inches="tight", facecolor="w")
    if show:
        plt.show()
    return fig


def plot_clockplot(series, label="", band="", scale=1e9, unit="nm", save=None, show=True):
    """Polar clock: median noise by weekday and hour-of-day."""
    indexed = _multiindex_series(series)
    table = stack_wday_time(indexed, scale)
    closing_row = table.iloc[[0]].copy()
    closing_row.index = [len(table)]
    table = pd.concat([table, closing_row])
    table.index = radial_hours(len(table))

    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    table.plot(ax=ax, cmap="tab20")
    _clock_axes_style(ax, unit=unit)
    ax.set_rmax(np.nanpercentile(series, 95) * 1.5 * scale)
    ax.set_rmin(0)
    plt.suptitle("Day/Hour Median Noise %s - [%s] Hz" % (label, band), fontsize=14)
    plt.subplots_adjust(top=0.85)

    if save:
        fig.savefig(save, bbox_inches="tight", facecolor="w")
    if show:
        plt.show()
    return fig


def plot_dailyplot(series, label="", band="", scale=1e9, unit="nm", save=None, show=True):
    """Line plot: median noise by hour for each weekday."""
    indexed = _multiindex_series(series)
    ax = stack_wday_time(indexed, scale).plot(figsize=(14, 8), cmap="tab20")
    plt.title("Daily Noise Levels - %s" % label)
    plt.ylabel("Amplitude (%s)" % unit)
    plt.xlabel("Hour of day (local time)")
    plt.grid()
    plt.xlim(0, 23)
    plt.ylim(0, np.nanpercentile(series, 95) * 1.5 * scale)

    if save:
        ax.figure.savefig(save, bbox_inches="tight", facecolor="w")
    if show:
        plt.show()
    return ax.figure


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SeismoNoise:
    """
    Wrapper around a ``dataframe_RMS`` dict for easy plotting.

    Parameters
    ----------
    dataframe_RMS : dict
        ``{"NET.STA.LOC.CHA": pd.DataFrame}`` where each DataFrame has a UTC
        DatetimeIndex and columns named ``"fmin-fmax"`` (e.g. ``"4.0-14.0"``).
    """

    def __init__(self, dataframe_RMS):
        self.displacement_RMS = dataframe_RMS

    def _prepare(self, mseedid, band="4.0-14.0", resample="30Min", time_zone="Europe/Brussels"):
        """
        Return a localized, resampled Series.
        If mseedid ends with ``*``, Z/E/N channels are combined via quadratic sum.
        """
        df = self.displacement_RMS
        if mseedid.endswith("*"):
            prefix = mseedid[:-1]
            combined = None
            for o in "ZEN":
                key = prefix + o
                if key not in df:
                    continue
                s = df[key][band].resample(resample).mean()
                combined = s ** 2 if combined is None else combined + s ** 2
            if combined is None:
                raise KeyError("No channels found for %s" % mseedid)
            series = combined ** 0.5
        else:
            series = df[mseedid][band].resample(resample).mean()
        return localize_tz_and_reindex(series, resample, time_zone)

    def plot(self, type="timeseries", band="4.0-14.0", mseedids=None,
             scale=1e9, unit="nm", time_zone="Europe/Brussels",
             save=None, fmt="pdf", show=True, resample="30Min"):
        """
        Plot noise data for all (or selected) channels.

        type : "timeseries" | "clockmaps" | "gridmaps" | "clockplots" | "dailyplots" | "*"
        """
        keys = mseedids if mseedids is not None else list(self.displacement_RMS.keys())

        for mseedid in keys:
            series = self._prepare(mseedid, band=band, resample=resample, time_zone=time_zone)

            def _save(suffix, _id=mseedid):
                return None if save is None else "%s%s-%s%s.%s" % (save, _id, band, suffix, fmt)

            kw = dict(label=mseedid, band=band, scale=scale, unit=unit, show=show)

            if type in ("*", "all", "timeseries"):
                plot_timeseries(series, save=_save(""), **kw)
            if type in ("*", "all", "clockmaps"):
                plot_clockmap(series, save=_save("-clockmap"), **kw)
            if type in ("*", "all", "gridmaps"):
                plot_gridmap(series, save=_save("-gridmap"), **kw)
            if type in ("*", "all", "clockplots"):
                plot_clockplot(series, save=_save("-clockplot"), **kw)
            if type in ("*", "all", "dailyplots"):
                plot_dailyplot(series, save=_save("-dailyplot"), **kw)

    # Convenience shortcuts
    def timeseries(self, **kw):  self.plot(type="timeseries", **kw)
    def clockmap(self, **kw):    self.plot(type="clockmaps", **kw)
    def gridmap(self, **kw):     self.plot(type="gridmaps", **kw)
    def clockplot(self, **kw):   self.plot(type="clockplots", **kw)
    def dailyplot(self, **kw):   self.plot(type="dailyplots", **kw)
