# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../")))
from pathlib import Path
from google_drive_downloader import GoogleDriveDownloader as gdd
import re
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")
from scripts._helper import mock_snakemake, update_config_from_wildcards, create_logger, PYPSA_EARTH_DIR


logger = create_logger(__name__)


def download_and_unzip_gdrive(config, disable_progress=False):
    """
        Downloads and unzips USA cutouts to submodules/pypsa-earth/cutouts folder 
    """
    resource = config["category"]
    file_path = os.path.join(PYPSA_EARTH_DIR, "tempfile.zip")
    destination = os.path.join(PYPSA_EARTH_DIR, config["destination"])
    url = config["urls"]["gdrive"]

    # retrieve file_id from path
    try:
        # cut the part before the ending \view
        partition_view = re.split(r"/view|\\view", str(url), 1)
        if len(partition_view) < 2:
            logger.error(
                f'Resource {resource} cannot be downloaded: "\\view" not found in url {url}'
            )
            return False

        # split url to get the file_id
        code_split = re.split(r"\\|/", partition_view[0])

        if len(code_split) < 2:
            logger.error(
                f'Resource {resource} cannot be downloaded: character "\\" not found in {partition_view[0]}'
            )
            return False

        # get file id
        file_id = code_split[-1]

        # remove tempfile.zip if exists
        Path(file_path).unlink(missing_ok=True)

        # download file from google drive
        gdd.download_file_from_google_drive(
            file_id=file_id,
            dest_path=file_path,
            showsize=not disable_progress,
            unzip=False,
        )
        with ZipFile(file_path, "r") as zipObj:
            bad_file = zipObj.testzip()
            if bad_file:
                logger.info(f"Corrupted file found: {bad_file}")
            else:
                logger.info("No errors found in the zip file.")
            # Extract all the contents of zip file in current directory
            zipObj.extractall(path=destination)
        # remove tempfile.zip
        Path(file_path).unlink(missing_ok=True)

        logger.info(f"Download resource '{resource}' from cloud '{url}'.")

        return True
    
    except Exception as e:
        logger.error(f"Failed to download or extract the file: {str(e)}")
        return False


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "retrieve_cutouts",
            configfile="configs/calibration/config.usa_PE.yaml",
            countries=["US"]
        )
    # update config based on wildcards
    config = update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # load cutouts configuration
    config_cutouts = config["custom_databundles"]["bundle_cutouts_USA"]

    # download cutouts
    downloaded = download_and_unzip_gdrive(config_cutouts)
