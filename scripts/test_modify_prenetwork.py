# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import pypsa
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, PYPSA_EARTH_DIR


logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "use_osm_data",
            configfile="configs/calibration/config.usa_PE.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load prenetwork
    prenetwork = pypsa.Network(snakemake.input.prenetwork)

    # save "modified" network
    prenetwork.export_to_netcdf(snakemake.output.network)
