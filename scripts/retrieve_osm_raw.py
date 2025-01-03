# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, \
                            download_and_unzip_gdrive, PYPSA_EARTH_DIR


logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_osm_raw",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load osm raw configuration
    config_osm_raw = config["custom_databundles"]["bundle_osm_raw_USA"]

    # destination for osm/raw
    destination = os.path.join(PYPSA_EARTH_DIR, snakemake.params.destination, "osm")

    # download osm/raw
    downloaded = download_and_unzip_gdrive(config_osm_raw,
                                           destination=destination,
                                           logger=logger)