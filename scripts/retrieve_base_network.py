# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from scripts._helper import (
    mock_snakemake,
    update_config_from_wildcards,
    create_logger,
    download_and_unzip_gdrive,
    download_and_unzip_zenodo,
    configure_logging,
)


logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_base_network",
            configfile="configs/calibration/config.base_AC.yaml",
        )

    configure_logging(snakemake)

    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load base.nc configuration
    config_base_network = config["custom_databundles"]["bundle_base_network_USA"]

    # Destination for base.nc
    output_path = Path(snakemake.output[0])
    destination = output_path.parent
    destination.mkdir(parents=True, exist_ok=True)

    # Download base.nc
    if "zenodo" in config_base_network["urls"]:
        download_and_unzip_zenodo(
            config_base_network,
            destination,
            logger,
        )
    else:
        download_and_unzip_gdrive(
            config_base_network,
            destination,
            logger,
        )

    if not output_path.exists():
        nc_files = list(destination.glob("*.nc"))

        if len(nc_files) == 1:
            nc_files[0].rename(output_path)
        elif len(nc_files) == 0:
            raise FileNotFoundError(
                f"No .nc file found in {destination} after downloading base network."
            )
        else:
            raise RuntimeError(
                f"Multiple .nc files found in {destination}: {nc_files}. "
                f"Cannot decide which one to use as {output_path}."
            )
