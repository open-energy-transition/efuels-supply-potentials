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
            "retrieve_osm_network",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load base_network configuration
    config_osm_network = config["custom_databundles"]["bundle_osm_network_USA"]

    # destination for base_network/
    destination = os.path.join(PYPSA_EARTH_DIR, snakemake.params.destination)

    # download base_network/
    downloaded = download_and_unzip_gdrive(config_osm_network,
                                           destination=destination,
                                           logger=logger)