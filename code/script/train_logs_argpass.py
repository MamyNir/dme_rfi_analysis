"""
RFI Analysis Module

This script loads observational data, filters a frequency band,
and computes RFI probability over time.

Features
--------
- Command-line interface using argparse
- Optional YAML configuration file
- Structured logging
- Input validation
"""

import os
import xarray as xr
import argparse
import logging
from typing import Optional, Dict

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml


# ----------------------------- #
# Logging Configuration
# ----------------------------- #
def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging format and level.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# ----------------------------- #
# YAML Configuration Loader
# ----------------------------- #
def load_config(config_path: Optional[str]) -> Dict:
    """
    Load parameters from a YAML configuration file.

    Parameters
    ----------
    config_path : str or None
        Path to YAML file

    Returns
    -------
    dict
        Configuration dictionary
    """
    if config_path is None:
        return {}

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    logging.info(f"Loaded configuration from {config_path}")
    return config


# ----------------------------- #
# Input Validation Functions
# ----------------------------- #
def validate_directory(path: str) -> str:
    """
    Validate that a path exists and is a directory.

    Parameters
    ----------
    path : str

    Returns
    -------
    str
        Validated directory path
    """
    path = os.path.expanduser(path)

    if not path:
        raise ValueError("Path cannot be empty.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    if not os.path.isdir(path):
        raise NotADirectoryError(f"Not a directory: {path}")

    return path


def validate_frequency(value: float, name: str) -> float:
    """
    Validate frequency input.

    Parameters
    ----------
    value : float
    name : str

    Returns
    -------
    float
    """
    if value is None:
        raise ValueError(f"{name} cannot be None.")

    if value <= 0:
        raise ValueError(f"{name} must be positive.")

    return value


# ----------------------------- #
# Core RFI Function
# ----------------------------- #
def rfi(data, freq_min: float, freq_max: float, time_resolution: int) -> pd.DataFrame:
    """
    Extract RFI probability in a frequency band and return time evolution.

    Parameters
    ----------
    data : xarray.DataArray
        Must contain dimensions: (time, frequency)
    freq_min : float
        Lower frequency bound (MHz)
    freq_max : float
        Upper frequency bound (MHz)
    time_resolution : int
        Time resolution in minutes

    Returns
    -------
    pandas.DataFrame
        RFI probability statistics over time
    """

    logging.info("Starting RFI computation")

    # Convert MHz → Hz
    band = data.sel(frequency=slice(freq_min * 1e6, freq_max * 1e6))

    # Validate frequency range
    min_freq = float(data.frequency[0] * 1e-6)
    max_freq = float(data.frequency[-1] * 1e-6)

    if not (freq_min >= min_freq and freq_max <= max_freq):
        raise ValueError(
            f"Frequency range must be within [{min_freq}, {max_freq}] MHz"
        )

    # Collapse over frequency
    rfi_time = band.mean(dim="frequency")

    # Convert to DataFrame
    df = rfi_time.to_dataframe()
    df.columns = ["probability"]
    df.index = pd.to_datetime(df.index, unit="s", utc=True)

    logging.debug("Data converted to DataFrame")

    # Resampling
    try:
        resampled = df.resample(f"{abs(time_resolution)}min").agg(
            {"probability": ["mean", "min", "max", "median"]}
        )
        logging.info("Resampling completed successfully")
        return resampled

    except Exception as e:
        logging.error(f"Resampling failed: {e}")
        raise


# ----------------------------- #
# Argument Parser
# ----------------------------- #
def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="RFI Analysis Tool")

    parser.add_argument("--path", type=str, help="Path to data directory")
    parser.add_argument("--freq-min", type=float, help="Minimum frequency (MHz)")
    parser.add_argument("--freq-max", type=float, help="Maximum frequency (MHz)")
    parser.add_argument("--time-resolution", type=int, default=60,
                        help="Time resolution in minutes")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    parser.add_argument("--output", type=str, required=True, 
                        help="Path to output CSV file")

    return parser.parse_args()


def plot(data):
    plt.figure(figsize=(6,4))
    plt.step(data.index, data["probability"]["max"], where="mid")
    plt.xlabel("Time (UTC)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.ylabel("Probability")
    plt.title("DME Hourly RFI Evolution")
    plt.grid(alpha=0.3)
    plt.savefig("dme.png")
    plt.show()


# ----------------------------- #
# Main Function
# ----------------------------- #
def main():
    """
    Main execution function.
    """
    args = parse_args()
    setup_logging(args.log_level)

    # Load YAML config if provided
    config = load_config(args.config)

    # Merge CLI + YAML (CLI overrides YAML)
    path = args.path or config.get("path")
    freq_min = args.freq_min or config.get("freq_min")
    freq_max = args.freq_max or config.get("freq_max")
    time_resolution = args.time_resolution or config.get("time_resolution", 60)

    # Validate inputs
    path = validate_directory(path)
    freq_min = validate_frequency(freq_min, "Minimum frequency")
    freq_max = validate_frequency(freq_max, "Maximum frequency")

    logging.info(f"Using data path: {path}")
    logging.info(f"Frequency range: {freq_min}–{freq_max} MHz")

    # TODO: Load your xarray dataset here
    data = xr.open_dataset(path)

    # Example call (data must be defined)
    result = rfi(data, freq_min, freq_max, time_resolution)
    result.to_csv(args.output)
    logging.info(f"Saved results to {args.output}")
    plot(result)


if __name__ == "__main__":
    '''
    python script.py --path ~/data --freq-min 100 --freq-max 200

    python script.py --config config.yaml
    
    '''
    main()
