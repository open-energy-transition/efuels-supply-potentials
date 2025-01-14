# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, create_logger


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "modify_aviation_demand",
            configfile="configs/calibration/config.base_AC.yaml",
        )


