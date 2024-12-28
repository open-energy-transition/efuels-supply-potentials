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


def shapes_outputs():
    outputs = [
        "shapes/country_shapes.geojson",
        "shapes/offshore_shapes.geojson",
        "shapes/africa_shape.geojson",
        "shapes/gadm_shapes.geojson"
    ]
    return outputs


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_shapes",
            configfile="configs/calibration/config.base_AC.yaml",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load shapes configuration
    config_shapes = config["custom_databundles"]["bundle_shapes_USA"]

    # destination for shapes
    destination = os.path.join(PYPSA_EARTH_DIR, snakemake.params.destination)

    # download shapes
    downloaded = download_and_unzip_gdrive(config_shapes,
                                           destination=destination,
                                           logger=logger)