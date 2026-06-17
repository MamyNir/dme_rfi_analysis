"""
RFI Monthly Evolution Analysis
================================
Analyzes the evolution of Radio Frequency Interference (RFI) probability
from a 3D xarray DataArray with dimensions: observation_date (UTC), time (hours), frequency.

Expected DataArray structure:
    - Dimensions: (date, time, frequency)  ← rename to match your actual dim names below
    - Coordinates:
        - date: datetime64 values (2019–2024)
        - time: float hours of day (0–24)
        - frequency: float MHz values
    - Values: RFI probability [0, 1]
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION  ← adjust these to match your DataArray
# ─────────────────────────────────────────────────────────────────
DATE_DIM      = "date"          # name of the observation-date dimension
TIME_DIM      = "time"          # name of the hour-of-day dimension
FREQ_DIM      = "frequency"     # name of the frequency dimension

YEARS         = list(range(2019, 2025))   # 2019 → 2024
MONTH_NAMES   = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]

CMAP_HEATMAP  = "inferno"       # colormap for freq–time heatmaps
CMAP_LINE     = "plasma"        # colormap for line plots
THRESHOLD     = 0.3             # probability threshold for "flagged" classification


# ─────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATA (replace this block with your real DataArray)
# ─────────────────────────────────────────────────────────────────
def make_synthetic_data():
    """Creates a realistic synthetic RFI dataset for demonstration."""
    rng = np.random.default_rng(42)

    dates = pd.date_range("2019-01-01", "2024-12-31", freq="7D")  # weekly obs
    times = np.arange(0, 24, 0.5)                                   # 30-min bins
    freqs = np.arange(100, 1500, 10, dtype=float)                   # 100–1490 MHz

    # Base RFI: higher at certain frequencies and daytime hours
    freq_base  = 0.05 + 0.3 * np.exp(-((freqs - 900) ** 2) / (2 * 200 ** 2))
    time_base  = 0.1  + 0.2 * np.sin(np.pi * times / 12) ** 2
    month_mod  = np.array([0.8,0.85,0.9,1.0,1.1,1.2,1.2,1.1,1.0,0.9,0.85,0.8])

    data = np.zeros((len(dates), len(times), len(freqs)), dtype=np.float32)
    for i, d in enumerate(dates):
        m = d.month - 1
        noise = rng.random((len(times), len(freqs))) * 0.1
        data[i] = np.clip(
            (freq_base[None, :] + time_base[:, None] + noise) * month_mod[m], 0, 1
        )

    da = xr.DataArray(
        data,
        dims=[DATE_DIM, TIME_DIM, FREQ_DIM],
        coords={DATE_DIM: dates, TIME_DIM: times, FREQ_DIM: freqs},
        name="rfi_probability",
        attrs={"units": "probability", "description": "RFI probability [0,1]"},
    )
    return da


# ─────────────────────────────────────────────────────────────────
# 2.  LOAD / PREPARE DATA
# ─────────────────────────────────────────────────────────────────
def load_data(da: xr.DataArray) -> xr.DataArray:
    """Ensure date coordinate is datetime64 and add helper coords."""
    if not np.issubdtype(da[DATE_DIM].dtype, np.datetime64):
        da[DATE_DIM] = pd.to_datetime(da[DATE_DIM].values)
    da = da.assign_coords(
        year  =(DATE_DIM, pd.DatetimeIndex(da[DATE_DIM].values).year),
        month =(DATE_DIM, pd.DatetimeIndex(da[DATE_DIM].values).month),
    )
    return da


# ─────────────────────────────────────────────────────────────────
# 3.  AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────
def monthly_mean_by_year(da: xr.DataArray) -> dict:
    """
    Returns a nested dict:
        result[year][month] = DataArray(time, frequency)  ← mean over obs dates
    """
    result = {}
    for yr in YEARS:
        result[yr] = {}
        yr_mask = da["year"] == yr
        da_yr = da.isel({DATE_DIM: yr_mask})
        for mo in range(1, 13):
            mo_mask = da_yr["month"] == mo
            da_mo = da_yr.isel({DATE_DIM: mo_mask})
            if da_mo.sizes[DATE_DIM] == 0:
                result[yr][mo] = None
            else:
                result[yr][mo] = da_mo.mean(dim=DATE_DIM)
    return result


def overall_monthly_mean(da: xr.DataArray) -> dict:
    """
    Mean over ALL years per calendar month → dict[month] = DataArray(time, frequency)
    """
    result = {}
    for mo in range(1, 13):
        mo_mask = da["month"] == mo
        da_mo = da.isel({DATE_DIM: mo_mask})
        result[mo] = da_mo.mean(dim=DATE_DIM) if da_mo.sizes[DATE_DIM] > 0 else None
    return result


def flagged_fraction(da: xr.DataArray, threshold: float = THRESHOLD) -> xr.DataArray:
    """
    For each (date), compute fraction of (time × frequency) pixels above threshold.
    Then group by year-month.
    """
    flagged = (da > threshold).mean(dim=[TIME_DIM, FREQ_DIM])   # shape: (date,)
    dates   = pd.DatetimeIndex(da[DATE_DIM].values)
    ym      = pd.PeriodIndex(dates, freq="M")
    df = pd.DataFrame({"ym": ym, "flagged": flagged.values})
    monthly = df.groupby("ym")["flagged"].mean().reset_index()
    monthly["year"]  = monthly["ym"].dt.year
    monthly["month"] = monthly["ym"].dt.month
    return monthly


# ─────────────────────────────────────────────────────────────────
# 4.  PLOT 1 – Frequency-averaged RFI vs Time, one line per year
#              subplot per calendar month  (12-panel figure)
# ─────────────────────────────────────────────────────────────────
def plot_monthly_time_evolution(monthly_by_year: dict, times: np.ndarray):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharey=True)
    fig.suptitle("RFI Probability – Time-of-Day Profile per Month\n(frequency-averaged)",
                 fontsize=15, fontweight="bold", y=1.01)

    cmap   = plt.get_cmap(CMAP_LINE, len(YEARS))
    colors = {yr: cmap(i) for i, yr in enumerate(YEARS)}

    for ax, mo in zip(axes.flat, range(1, 13)):
        for yr in YEARS:
            da_mo = monthly_by_year[yr][mo]
            if da_mo is None:
                continue
            profile = da_mo.mean(dim=FREQ_DIM).values
            ax.plot(times, profile, label=str(yr), color=colors[yr], lw=1.4, alpha=0.85)

        ax.set_title(MONTH_NAMES[mo - 1], fontweight="bold")
        ax.set_xlabel("Hour (UTC)")
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(6))
        ax.xaxis.set_minor_locator(MultipleLocator(3))
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.axhline(THRESHOLD, color="red", ls=":", lw=0.8, alpha=0.6)

    # Legend outside
    handles = [plt.Line2D([0],[0], color=colors[yr], lw=2, label=str(yr)) for yr in YEARS]
    fig.legend(handles=handles, title="Year", loc="upper right",
               bbox_to_anchor=(1.02, 0.98), framealpha=0.9)

    axes[0, 0].set_ylabel("Mean RFI probability")
    axes[1, 0].set_ylabel("Mean RFI probability")
    axes[2, 0].set_ylabel("Mean RFI probability")

    plt.tight_layout()
    plt.savefig("rfi_time_profile_by_month.png", dpi=150, bbox_inches="tight")
    print("Saved: rfi_time_profile_by_month.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# 5.  PLOT 2 – Frequency–Time heatmap, one panel per month
#              (averaged across all years)
# ─────────────────────────────────────────────────────────────────
def plot_freq_time_heatmaps(overall_monthly: dict, times: np.ndarray, freqs: np.ndarray):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True, sharey=True)
    fig.suptitle("Mean RFI Probability – Frequency vs Time-of-Day\n(all years combined)",
                 fontsize=15, fontweight="bold", y=1.01)

    vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = None
    for ax, mo in zip(axes.flat, range(1, 13)):
        da_mo = overall_monthly[mo]
        if da_mo is None:
            ax.set_title(MONTH_NAMES[mo - 1])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # da_mo has dims (time, frequency)
        grid = da_mo.values   # shape (n_time, n_freq)
        im = ax.pcolormesh(times, freqs / 1e3, grid.T,
                           cmap=CMAP_HEATMAP, norm=norm, shading="auto")
        ax.set_title(MONTH_NAMES[mo - 1], fontweight="bold")

    for ax in axes[2]:
        ax.set_xlabel("Hour (UTC)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency (GHz)")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                            fraction=0.015, pad=0.02, label="RFI Probability")
        cbar.ax.yaxis.label.set_fontsize(11)

    plt.tight_layout()
    plt.savefig("rfi_freq_time_heatmap_by_month.png", dpi=150, bbox_inches="tight")
    print("Saved: rfi_freq_time_heatmap_by_month.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# 6.  PLOT 3 – Flagged fraction time-series (monthly resolution)
# ─────────────────────────────────────────────────────────────────
def plot_flagged_fraction_timeseries(da: xr.DataArray):
    df = flagged_fraction(da)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_title(f"Fraction of (time × frequency) bins with RFI > {THRESHOLD:.0%}",
                 fontsize=13, fontweight="bold")

    for yr in YEARS:
        sub = df[df["year"] == yr].sort_values("month")
        x   = sub["month"].values
        y   = sub["flagged"].values
        ax.plot(x + (yr - 2019) * 0.1, y, "o-", label=str(yr), lw=1.5, ms=4)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("Flagged fraction")
    ax.set_ylim(0, None)
    ax.grid(True, ls="--", alpha=0.35)
    ax.legend(title="Year", loc="upper right", ncol=3)

    plt.tight_layout()
    plt.savefig("rfi_flagged_fraction_timeseries.png", dpi=150, bbox_inches="tight")
    print("Saved: rfi_flagged_fraction_timeseries.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# 7.  PLOT 4 – Seasonal cycle: mean RFI per month  ± std across years
# ─────────────────────────────────────────────────────────────────
def plot_seasonal_cycle(da: xr.DataArray):
    """Overall mean RFI probability per calendar month with inter-annual spread."""
    means, stds = [], []
    for mo in range(1, 13):
        yr_means = []
        for yr in YEARS:
            mask = (da["month"] == mo) & (da["year"] == yr)
            subset = da.isel({DATE_DIM: mask})
            if subset.sizes[DATE_DIM] > 0:
                yr_means.append(float(subset.mean()))
        means.append(np.mean(yr_means) if yr_means else np.nan)
        stds.append(np.std(yr_means) if yr_means else np.nan)

    means = np.array(means)
    stds  = np.array(stds)
    months = np.arange(1, 13)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(months, means, "o-", color="#E63946", lw=2.2, ms=7, label="Inter-annual mean")
    ax.fill_between(months, means - stds, means + stds,
                    alpha=0.25, color="#E63946", label="±1 std (year-to-year)")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_ylabel("Mean RFI probability")
    ax.set_title("Seasonal Cycle of RFI (2019–2024)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, None)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig("rfi_seasonal_cycle.png", dpi=150, bbox_inches="tight")
    print("Saved: rfi_seasonal_cycle.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    # ── Load your data ──────────────────────────────────────────
    # Replace the line below with your actual DataArray, e.g.:
    #   da = xr.open_dataarray("rfi_data.nc")
    da = make_synthetic_data()
    print(f"DataArray shape : {da.shape}")
    #print(f"Dimensions      : {dict(da.dims)}")
    print(f"Date range      : {str(da[DATE_DIM].values[0])[:10]} → "
          f"{str(da[DATE_DIM].values[-1])[:10]}")
    print()

    da = load_data(da)

    times = da[TIME_DIM].values
    freqs = da[FREQ_DIM].values

    # ── Aggregate ────────────────────────────────────────────────
    print("Computing monthly aggregates …")
    monthly_by_year  = monthly_mean_by_year(da)
    overall_monthly  = overall_monthly_mean(da)

    # ── Plot ─────────────────────────────────────────────────────
    print("Plotting …")
    plot_monthly_time_evolution(monthly_by_year, times)
    plot_freq_time_heatmaps(overall_monthly, times, freqs)
    plot_flagged_fraction_timeseries(da)
    plot_seasonal_cycle(da)

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()