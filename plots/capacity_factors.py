# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))
from state_analysis import get_state_mapping

from scripts._helper import (
    DATA_DIR,
    PLOTS_DIR,
    build_directory,
    load_network,
    mock_snakemake,
    update_config_from_wildcards,
)

warnings.filterwarnings("ignore")


def get_capacity_factor(n, alternative_clustering):
    # get capacity factors
    carriers = ["solar", "onwind", "offwind-ac", "offwind-dc"]
    capacity_factors = {}
    average_CF = {}

    for c in carriers:
        res_gens = n.generators.query("carrier in @c").index
        capacity_factor = n.generators_t.p_max_pu[res_gens]
        capacity_factor.columns = capacity_factor.columns.map(
            lambda x: x.split("_AC")[0]
        )
        if alternative_clustering:
            capacity_factor.columns = capacity_factor.columns.map(gadm_state)
        if np.any(pd.isna(capacity_factor.columns)):
            capacity_factor = capacity_factor.drop(columns=[np.nan])
        capacity_factor = capacity_factor.sort_index(axis=1)
        capacity_factor.loc[:, "USA"] = capacity_factor.mean(axis=1)
        capacity_factor.loc["Annual average", :] = capacity_factor.mean(axis=0)
        capacity_factors[c] = capacity_factor
        average_CF[c] = capacity_factor.loc[:, "USA"].mean()

    return capacity_factors, average_CF


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "get_capacity_factor",
            configfile="configs/calibration/config.base_AC.yaml",
            simpl="",
            ll="copt",
            opts="Co2L-24H",
            clusters="10",
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # get alternative clustering
    alternative_clustering = snakemake.params.alternative_clustering

    # Plots directory
    build_directory(PLOTS_DIR)

    # load the network
    n = load_network(snakemake.input.unsolved_network)

    # get gadm state mapping
    gadm_state = get_state_mapping(snakemake.input.gadm)

    # retrieve capacity factors
    capacity_factors, average_CF = get_capacity_factor(n, alternative_clustering)

    # Create an Excel writer
    with pd.ExcelWriter(snakemake.output.capacity_factors, engine="openpyxl") as writer:
        for carrier, data in capacity_factors.items():
            # Add the average_CF as a new column to the DataFrame
            data["Average_CF"] = [average_CF[carrier]] + [None] * (data.shape[0] - 1)

            # Write to a separate sheet named after the carrier
            data.to_excel(writer, sheet_name=carrier, index=True)

    logging.info(f"Data written to {snakemake.output.capacity_factors}")
