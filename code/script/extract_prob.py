"""RFI Analysis Module

This script loads observational data, filters a frequency band,
and computes RFI probability over given dimension.

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
import yaml
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict

# ----------------------------- #
# Logging Configuration
# ----------------------------- #
def setup_logging(file, level: str = "INFO") -> None:
    """
    Configure logging format and level.

    Parameters
    ----------
    file : str
        Name of the log output file
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(file, mode="w"),
            logging.StreamHandler()
        ]
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
def rfi(data, freq_min: float, freq_max: float, dimension: list) -> xr.DataArray:
    """
    Extract RFI probability of RFI from the master an counter array in a given dimension.

    Parameters
    ----------
    data : xarray.Datasets
        Must contain counter and master in 5D (time, frequency, baseline, azimuth, elevation) 
    freq_min : float
        Lower frequency bound (MHz)
    freq_max : float
        Upper frequency bound (MHz)
    dimension : list
        The axis to get the probability

    Returns
    -------
    xarray.dataArray
        RFI probability as a function of given dimension
    """

    # Load xarray.Datasets
    counter = data["counter"]
    master = data["master"]

    # Validate frequency range
    min_value = float(data.frequency[0] * 1e-6)
    max_value = float(data.frequency[-1] * 1e-6)

    if not (freq_min >= min_value and freq_max <= max_value):
        raise ValueError(
            f"Frequency range must be within [{min_value}, {max_value}] MHz"
        )

    # Select frequency to compute
    sel_counter = counter.sel(frequency=slice(freq_min * 1e6, freq_max * 1e6))
    sel_master = master.sel(frequency=slice(freq_min * 1e6, freq_max * 1e6))

    # Accessing probability
    axis = ("time", "frequency", "baseline", "azimuth", "elevation")
    new_axis = [ x for x in axis if x not in dimension]

    sum_counter = sel_counter.sum(dim=new_axis)
    sum_master = sel_master.sum(dim=new_axis)

    probability = sum_master / sum_counter
    logging.debug(f"Extracting probability in {dimension}")

    return probability


def process_file(args):
    """
    Process multiple file at the same time and write the result in a zarr file

    Parameters
    ----------
    file : str
        File name of the file to process

    Returns
    -------
    None

    """
    file, path, freq_min, freq_max, dimension, initial, band = args
    full_path = os.path.join(path, file, "arr")

    try:
        logging.info(f"Processing {file}")

        data = xr.open_zarr(full_path, consolidated=False)
        probability = rfi(data, freq_min, freq_max, dimension)

        # Saving dataset into zarr file
        ds = probability.to_dataset(name= "probability")
        file_name = f"{file[:-5]}_prob_{initial}.zarr"
        dirr_name = f"{initial}_{int(freq_min)}_{int(freq_max)}"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirr_result = os.makedirs(os.path.join(
            root,
            "results",
            band,
            "probability",
            dirr_name,
            file_name),
            exist_ok=True
            )
        ds.to_zarr(dirr_result, mode="w")

    except Exception as e:

        logging.warning(f"Skipping {full_path}: {e}")


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
    parser.add_argument("--band", type=str, default="lband", help="Type of the frequency band", required=False)
    parser.add_argument("--dim", nargs="+", default=["time"], type=str, help="Dimension of the probability to compute")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")

    return parser.parse_args()


# ----------------------------- #
# Main Function
# ----------------------------- #
def main():
    """
    Main execution function.
    """
    args = parse_args()

    # Load YAML config if provided
    config = load_config(args.config)

    # Merge CLI + YAML (CLI overrides YAML)
    path = args.path or config.get("path")
    freq_min = args.freq_min or config.get("freq_min")
    freq_max = args.freq_max or config.get("freq_max")
    band = args.band or config.get("band")
    dimension = args.dim or config.get("dim", "time")

    # Convert "str" or nested list dimension from argparse to list
    if isinstance(dimension, str):
        dimension = [dimension]
    if len(dimension) == 1 and isinstance(dimension[0], list):
        dimension = dimension[0]

    # Get initial of dimension
    abbr = {
    "azimuth": "azim",
    "elevation": "elev",
    "time": "time",
    "frequency": "freq",
    "baseline": "basel"}

    initial = "_".join(abbr.get(d, d) for d in dimension)

    # Validate inputs
    freq_min = validate_frequency(freq_min, "Minimum frequency")
    freq_max = validate_frequency(freq_max, "Maximum frequency")

    log_name = f".{initial}_{int(freq_min)}_{int(freq_max)}.log"
    setup_logging(log_name, args.log_level)

    logging.info(f"Frequency range: {int(freq_min)}–{(freq_max)} MHz")

    path = validate_directory(path)

    logging.info(f"Using data path: {path}")

    # Extract and save probability from zarr file
    zarr_file = [
        item for item in os.listdir(path)
        if os.path.isdir(os.path.join(path, item))
        and item.endswith(".zarr")
    ]

    logging.info("Starting RFI computation")
    tasks = [(file, path, freq_min, freq_max, dimension, initial, band) for file in zarr_file]

    with ProcessPoolExecutor(max_workers=4) as executor:
        list(executor.map(process_file, tasks))



if __name__ == "__main__":
    '''
    python script.py --path ~/data --freq-min 100 --freq-max 200 --dim time frequency --band lband

    python script.py --config config.yaml

    '''
    main()

