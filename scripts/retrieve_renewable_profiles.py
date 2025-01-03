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
            "retrieve_renewable_profiles",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load renewable_profiles configuration
    config_renewable_profiles = config["custom_databundles"]["bundle_renewable_profiles_USA"]

    # destination for renewable_profiles/
    destination = os.path.join(PYPSA_EARTH_DIR, snakemake.params.destination)

    # url for alternative or voronoi clustering
    if snakemake.params.alternative_clustering:
        url = config_renewable_profiles["urls"]["alternative_clustering"]
    else:
        url = config_renewable_profiles["urls"]["voronoi_clustering"]

    # download base_network/
    downloaded = download_and_unzip_gdrive(config_renewable_profiles,
                                           destination=destination,
                                           logger=logger,
                                           url=url)