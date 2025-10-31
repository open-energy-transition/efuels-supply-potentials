# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
import warnings

from scripts._helper import (
    configure_logging,
    create_logger,
    download_and_unzip_gdrive,
    mock_snakemake,
    update_config_from_wildcards,
)

warnings.filterwarnings("ignore")
logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_demand_data",
            configfile="configs/calibration/config.base.yaml",
        )

    configure_logging(snakemake)

    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load demand data configuration
    config_demand_data = config["custom_databundles"]["bundle_demand_data_USA"]

    # destination for demand data
    destination = "data"

    # download demand data
    downloaded = download_and_unzip_gdrive(
        config_demand_data, destination=destination, logger=logger
    )
