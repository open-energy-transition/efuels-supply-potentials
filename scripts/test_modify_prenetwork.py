# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
import warnings
import pypsa

from scripts._helper import (
    create_logger,
    mock_snakemake,
    update_config_from_wildcards,
)

warnings.filterwarnings("ignore")
logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "test_modify_network",
            configfile="configs/calibration/config.base.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load prenetwork
    prenetwork = pypsa.Network(snakemake.input.prenetwork)

    # save "modified" network
    prenetwork.export_to_netcdf(snakemake.output.network)
