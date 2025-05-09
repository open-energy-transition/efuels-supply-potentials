# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import shutil
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
warnings.filterwarnings("ignore")

from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, configure_logging

logger = create_logger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "prepare_growth_factors",
            configfile="configs/calibration/config.base.yaml",
        )

    configure_logging(snakemake)

    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # scenario: High / Medium
    scenario = config["demand_projection"]["scenario"]
    source_file = f"data/US_growth_rates/{scenario}/growth_factors_cagr.csv"
    target_file = PYPSA_EARTH_DIR + "data/demand/growth_factors_cagr.csv"

    os.makedirs(os.path.dirname(target_file), exist_ok=True)

    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        logger.info(f"Copied {source_file} to {target_file}")
    else:
        logger.warning(f"{source_file} does not exist.")
