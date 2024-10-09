# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
from pathlib import Path
import yaml
import pypsa
import logging
import snakemake as sm
from pypsa.descriptors import Dict
from snakemake.script import Snakemake
import warnings
warnings.filterwarnings("ignore")


# get the base working directory
BASE_PATH = os.path.abspath(os.path.join(__file__, "../.."))
PLOTS_DIR = BASE_PATH + "/plots/results/"
DATA_DIR = BASE_PATH + "/data/"
# get pypsa-earth submodule directory path
PYPSA_EARTH_DIR = BASE_PATH + "/submodules/pypsa-earth"

LINE_OPTS = {"2021": "copt"}


def load_network(filepath):
    """ Input:
            filepath - full path to the network
        Output:
            n - PyPSA network
    """
    try:
        n = pypsa.Network(filepath)
        logging.info(f"Loading {filepath}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    return n


def mock_snakemake(rule_name, configfile=None, **wildcards):
    """
    This function is expected to be executed from the "scripts"-directory of "
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards**.

    Parameters
    ----------
    rule_name: str
        name of the rule for which the snakemake object should be generated
    wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """

    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir.parent)
    for p in sm.SNAKEFILE_CHOICES:
        if Path(p).exists():
            snakefile = p
            break
    if isinstance(configfile, str):
        with open(configfile, 'r') as file:
            configfile = yaml.safe_load(file)

    workflow = sm.Workflow(
        snakefile, overwrite_configfiles=[], rerun_triggers=[], overwrite_config=configfile
    )  # overwrite_config=config
    workflow.include(snakefile)
    workflow.global_resources = {}
    try:
        rule = workflow.get_rule(rule_name)
    except Exception as exception:
        print(
            exception,
            f"The {rule_name} might be a conditional rule in the Snakefile.\n"
            f"Did you enable {rule_name} in the config?",
        )
        raise
    dag = sm.dag.DAG(workflow, rules=[rule])
    wc = Dict(wildcards)
    job = sm.jobs.Job(rule, dag, wc)

    def make_accessable(*ios):
        for io in ios:
            for i in range(len(io)):
                io[i] = Path(io[i]).absolute()

    make_accessable(job.input, job.output, job.log)
    snakemake = Snakemake(
        job.input,
        job.output,
        job.params,
        job.wildcards,
        job.threads,
        job.resources,
        job.log,
        job.dag.workflow.config,
        job.rule.name,
        None,
    )
    snakemake.benchmark = job.benchmark

    # create log and output dir if not existent
    for path in list(snakemake.log) + list(snakemake.output):
        build_directory(path)

    os.chdir(script_dir)
    return snakemake


def update_config_from_wildcards(config, w):
    if w.get("planning_horizon"):
        planning_horizon = w.planning_horizon
        config["validation"]["planning_horizon"] = planning_horizon
    if w.get("clusters"):
        clusters = w.clusters
        config["scenario"]["clusters"] = clusters
    if w.get("countries"):
        countries = w.countries
        config["scenario"]["countries"] = countries
    if w.get("simpl"):
        simpl = w.simpl
        config["scenario"]["simpl"] = simpl
    if w.get("opts"):
        opts = w.opts
        config["scenario"]["opts"] = opts
    if w.get("ll"):
        ll = w.ll
        config["scenario"]["ll"] = ll
    return config


def build_directory(path, just_parent_directory=True):
    """
    It creates recursively the directory and its leaf directories.

    Parameters:
        path (str): The path to the file
        just_parent_directory (Boolean): given a path dir/subdir
            True: it creates just the parent directory dir
            False: it creates the full directory tree dir/subdir
    """

    # Check if the provided path points to a directory
    if just_parent_directory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_solved_network_path(scenario_folder):
    """
    Get the full path to the PyPSA network file from the specified scenario folder.
    Assumes that only one network file exists in the folder.

    Args:
        scenario_folder (str): Folder containing the scenario data.

    Returns:
        str: Full path to the network file.
    """
    results_dir = os.path.join(
        BASE_PATH, f"submodules/pypsa-earth/results/{scenario_folder}/networks")
    filenames = os.listdir(results_dir)

    # Ensure only one network file exists
    if len(filenames) != 1:
        logging.warning(f"Only 1 network per scenario is allowed currently!")
    filepath = os.path.join(results_dir, filenames[0])

    return filepath


def load_pypsa_network(scenario_folder):
    """
    Load a PyPSA network from a specific scenario folder.

    Args:
        scenario_folder (str): Folder containing the scenario data.

    Returns:
        pypsa.Network: The loaded PyPSA network object.
    """
    network_path = get_solved_network_path(scenario_folder)
    network = pypsa.Network(network_path)
    return network


def load_network(path):
    """
    Loads a PyPSA network from a path
    """
    network = pypsa.Network(path)
    return network


def create_logger(logger_name, level=logging.INFO):
    """
    Create a logger for a module and adds a handler needed to capture in logs
    traceback from exceptions emerging during the workflow.
    """
    logger_instance = logging.getLogger(logger_name)
    logger_instance.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    logger_instance.addHandler(handler)
    return logger_instance
