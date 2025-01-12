# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import shutil
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger


logger = create_logger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_custom_powerplants",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    old_custom_powerplants_path = snakemake.input.old_path
    new_custom_powerplants_path = snakemake.output.destination

    with open(snakemake.output.powerplants_dummy_input, "w") as f:
        f.write("success")

    shutil.copy(old_custom_powerplants_path, new_custom_powerplants_path)
    logger.info(f"Retrieved custom_powerplants.csv file successfully")