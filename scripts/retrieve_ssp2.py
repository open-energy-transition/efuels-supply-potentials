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
            "retrieve_ssp2",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load ssp2 configuration
    config_ssp2 = config["custom_databundles"]["bundle_ssp2"]

    # destination for NorthAmerica.csv file
    destination = os.path.join(PYPSA_EARTH_DIR, snakemake.params.destination)

    # remove NorthAmerica.nc file
    nc_path = PYPSA_EARTH_DIR + "/data/ssp2-2.6/2030/era5_2013/NorthAmerica.nc"
    
    if os.path.isfile(nc_path):
        os.path.exists(nc_path)
        os.remove(nc_path)
        logger.info(f"Removed {nc_path} file successfully")

    # # download ssp2
    downloaded = download_and_unzip_gdrive(config_ssp2,
                                           destination=destination,
                                           logger=logger)