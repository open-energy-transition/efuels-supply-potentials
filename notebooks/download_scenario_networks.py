# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download solved PyPSA network files for scenarios from Google Drive.

Usage examples:
    # Download all files for scenario 2
    python download_scenario_networks.py --scenario-id 2
    
    # Download all files for scenarios 1, 3, and 5
    python download_scenario_networks.py --scenario-id 1 3 5
    
    # Download all scenarios
    python download_scenario_networks.py --all
    
    # Download specific years for scenario 2
    python download_scenario_networks.py --scenario-id 2 --years 2030 2035
    
    # Download with custom resolution (if different files exist)
    python download_scenario_networks.py --scenario-id 1 --resolution 1H
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from google_drive_downloader import GoogleDriveDownloader as gdd


# -----------------------------
# Google Drive Download Links
# -----------------------------
# Format: {scenario_id: {year: google_drive_url}}
# Replace placeholder URLs with actual Google Drive sharing links
NETWORK_FILES = {
    1: {
        # https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
        2030: "https://drive.google.com/file/d/17uM_AKQ4DNhjR6o0CPzBWzKgZZSA5SEe/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1ib991fC5Zlkns4hGFfhS4q5S0KCX9_6u/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/192C8wxKQnyYYTD79TJAV15JKtePD4dhg/view?usp=drive_link",
    },
    2: {
        2030: "https://drive.google.com/file/d/1nBEauWCj3_9H9Qdw46cNTejtIAIQ9TPb/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1qXWjkgV_s574Do17Qo_Q6mQ0XP6y54fH/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1oFpbDPqyenjiAHz7aB6zgSlYfXnJXOBo/view?usp=drive_link",
    },
    3: {
        2030: "https://drive.google.com/file/d/1aHk_Ry69W-_0DqB52VpnmMXONW9Zm2Lj/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/11S01QUC-8GlMeKJfyxYgwBjuMQRf1bl2/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/18EkTiqPG9xfLxDELzToF5Y_cLTzw4EW7/view?usp=drive_link",
    },
    4: {
        2030: "https://drive.google.com/file/d/1YND60RtMY2Bab2-RVBizHsvNp_p8SV8e/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1Tic6s-UpgVsT8hldcez3bygR-Fy4SSgv/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1c5aV-WdPzTiveUouZo1iiipcIg9SLdvH/view?usp=drive_link",
    },
    5: {
        2030: "https://drive.google.com/file/d/1VfDoy_R2hZIuzCekq37TLXbH4ON1-y69/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1bTHkjL3mjhXfWM9TpcLKvwonQ1DSFnp1/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1iu0KBrZgFVU-hR7u8IaCYx20XuU7m15J/view?usp=drive_link",
    },
    6: {
        2030: "PLACEHOLDER_URL_S06_2030",
        2035: "PLACEHOLDER_URL_S06_2035",
        2040: "PLACEHOLDER_URL_S06_2040",
    },
    7: {
        2030: "https://drive.google.com/file/d/1hpxdlopy70ggn7q0xDLe9OlXgWiCJt00/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1oMpKm-4Z4zjJmtDAo6dNBqIIpsE8m6RM/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/17nM3MHHYN4sFnFF9zczwEQH9m-Rkzhoh/view?usp=drive_link",
    },
    8: {
        2030: "https://drive.google.com/file/d/1RfSjUU6wnMdzklKb9KpHKBo14acPxvmB/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/16F27Fx5nCIrB7u4eFSneKoQwKNda9X8l/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1Gx_oZKAQZw06lYffpR6qxSpUaYQ8V6_I/view?usp=drive_link",
    },
    9: {
        2030: "https://drive.google.com/file/d/1kocL7Da4jGO9sW2XRaY3Zv9xYS5QfKZt/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1QJzgrIc-IJKLfLOBDgRsqjjxIG7vV7cS/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1OvXRx2mLAV1NHXINY5sw0fzw0Bkp3owT/view?usp=drive_link",
    },
    10: {
        2030: "https://drive.google.com/file/d/1-ydCHaJigd2EULFuJd3brtcqkyDt_U08/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1dWvAiEdy-ya-qPVq0omdePtYYLnEqDCB/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1_HChV3SdxIE-kXMt3NqBx3cfwQVMva04/view?usp=drive_link",
    },
}

# Network file naming patterns
# Scenarios 1, 5, 6 use 'lcopt' wildcard; others use 'lv1'
FILENAME_PATTERNS = {
    1: "elec_s_100_ec_lcopt_3H_3H_{year}_0.071_AB_10export.nc",
    2: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
    3: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
    4: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
    5: "elec_s_100_ec_lcopt_3H_3H_{year}_0.071_AB_10export.nc",
    6: "elec_s_100_ec_lcopt_3H_3H_{year}_0.071_AB_10export.nc",
    7: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
    8: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
    9: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
    10: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
}

# Scenario names for user feedback
SCENARIO_NAMES = {
    1: "Reference - No e-kerosene mandate",
    2: "Reference - ReFuel EU",
    3: "Reference - ReFuel EU+",
    4: "Reference - ReFuel EU-",
    5: "Sensitivity - High climate ambition & No e-kerosene mandate",
    6: "Sensitivity - High climate ambition & ReFuel EU",
    7: "Sensitivity - Optimistic electricity generation costs",
    8: "Sensitivity - Optimistic electrolyzer costs",
    9: "Sensitivity - Conservative electrolyzer costs",
    10: "Sensitivity - Biogenic point-source CO2 only",
}


# -----------------------------
# Helper Functions
# -----------------------------
def extract_file_id_from_url(url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from sharing URL.

    Supports formats:
    - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/uc?id=FILE_ID
    """
    if "PLACEHOLDER" in url:
        return None

    if "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        return url.split("id=")[1].split("&")[0]
    else:
        return None


def download_file_from_google_drive(file_id: str, destination: Path, filename: str) -> bool:
    """
    Download a file from Google Drive using file ID.

    Args:
        file_id: Google Drive file ID
        destination: Local directory path
        filename: Name for the downloaded file

    Returns:
        True if download successful, False otherwise
    """
    try:
        dest_path = destination / filename

        # Download using googledrivedownloader
        gdd.download_file_from_google_drive(
            file_id=file_id,
            dest_path=str(dest_path),
            unzip=False,
            showsize=True
        )

        # Verify the download was successful
        if dest_path.exists():
            file_size = dest_path.stat().st_size
            # Check if file is suspiciously small (likely an error page)
            if file_size < 10 * 1024 * 1024:  # Less than 10MB
                with open(dest_path, 'rb') as f:
                    header = f.read(1024)
                    if b'<html' in header.lower() or b'<!doctype html' in header.lower():
                        print(
                            f"      ❌ Downloaded file is HTML error page ({file_size / 1024:.1f} KB)")
                        print(f"         The file has restricted permissions.")
                        dest_path.unlink()  # Delete invalid file
                        return False
            return True
        return False

    except Exception as e:
        print(f"  Error downloading file: {e}")
        print(f"  💡 Tip: Make sure the file has proper sharing permissions")
        print(f"       (Should be set to 'Anyone with the link')")
        return False


def download_scenario_networks(
    scenario_ids: List[int],
    years: Optional[List[int]] = None,
    resolution: str = "3H",
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> Dict[int, Dict[int, bool]]:
    """
    Download network files for specified scenarios and years.

    Args:
        scenario_ids: List of scenario IDs (1-10)
        years: List of years to download (default: [2030, 2035, 2040])
        resolution: Temporal resolution (default: 3H)
        output_dir: Base output directory (default: results/scenarios)
        skip_existing: Skip download if file already exists

    Returns:
        Dictionary mapping scenario_id -> year -> download_success
    """
    if years is None:
        years = [2030, 2035, 2040]

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "scenarios"

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for scenario_id in scenario_ids:
        if scenario_id not in NETWORK_FILES:
            print(
                f"⚠️  Warning: Scenario {scenario_id} not found in configuration. Skipping.")
            continue

        print(f"\n{'='*80}")
        print(f"📦 Scenario {scenario_id:02d}: {SCENARIO_NAMES[scenario_id]}")
        print(f"{'='*80}")

        scenario_dir = output_dir / f"scenario_{scenario_id:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        results[scenario_id] = {}

        for year in years:
            if year not in NETWORK_FILES[scenario_id]:
                print(
                    f"  ⚠️  Year {year} not available for scenario {scenario_id}. Skipping.")
                results[scenario_id][year] = False
                continue

            # Generate filename
            filename = FILENAME_PATTERNS[scenario_id].format(year=year)
            if resolution != "3H":
                # Replace resolution in filename if different from default
                filename = filename.replace(
                    "3H_3H", f"{resolution}_{resolution}")

            destination = scenario_dir / filename

            # Check if file already exists
            if skip_existing and destination.exists():
                file_size = destination.stat().st_size / (1024**3)  # Size in GB
                print(
                    f"  ✓ {year}: Already exists ({file_size:.2f} GB) - Skipping")
                results[scenario_id][year] = True
                continue

            # Get Google Drive URL
            gdrive_url = NETWORK_FILES[scenario_id][year]

            # Check for placeholder
            if "PLACEHOLDER" in gdrive_url:
                print(
                    f"  ⚠️  {year}: No Google Drive URL configured (placeholder found)")
                print(f"      Please update the URL in NETWORK_FILES dictionary")
                results[scenario_id][year] = False
                continue

            # Extract file ID
            file_id = extract_file_id_from_url(gdrive_url)
            if not file_id:
                print(f"  ❌ {year}: Invalid Google Drive URL format")
                results[scenario_id][year] = False
                continue

            # Download file
            print(f"  ⬇️  {year}: Downloading {filename}...")
            success = download_file_from_google_drive(
                file_id, scenario_dir, filename)

            if success:
                file_size = destination.stat().st_size / (1024**3)  # Size in GB
                print(f"  ✓ {year}: Download complete ({file_size:.2f} GB)")
                results[scenario_id][year] = True
            else:
                print(f"  ❌ {year}: Download failed")
                results[scenario_id][year] = False

    return results


def print_summary(results: Dict[int, Dict[int, bool]]):
    """Print download summary statistics."""
    print(f"\n{'='*80}")
    print("📊 Download Summary")
    print(f"{'='*80}")

    total_files = 0
    successful = 0
    failed = 0

    for scenario_id, year_results in results.items():
        scenario_success = sum(year_results.values())
        scenario_total = len(year_results)

        total_files += scenario_total
        successful += scenario_success
        failed += scenario_total - scenario_success

        status = "✓" if scenario_success == scenario_total else "⚠️"
        print(
            f"{status} Scenario {scenario_id:02d}: {scenario_success}/{scenario_total} files downloaded")

    print(f"\n{'='*80}")
    print(f"Total: {successful}/{total_files} successful, {failed} failed")
    print(f"{'='*80}")


# -----------------------------
# CLI Interface
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download solved PyPSA network files from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all years for scenario 2
  python download_scenario_networks.py --scenario-id 2
  
  # Download multiple scenarios
  python download_scenario_networks.py --scenario-id 1 3 5 10
  
  # Download all scenarios
  python download_scenario_networks.py --all
  
  # Download specific years only
  python download_scenario_networks.py --scenario-id 2 --years 2030 2035
  
  # Download with custom resolution
  python download_scenario_networks.py --scenario-id 1 --resolution 1H
  
  # Specify output directory
  python download_scenario_networks.py --scenario-id 2 --output-dir /path/to/output
  
  # Force re-download existing files
  python download_scenario_networks.py --scenario-id 2 --force
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario-id",
        type=int,
        nargs="+",
        help="Scenario IDs to download (1-10). Can specify multiple IDs.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all scenarios (1-10)",
    )

    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2030, 2035, 2040],
        help="Years to download (default: 2030 2035 2040)",
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="3H",
        help="Temporal resolution (default: 3H)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for downloaded files (default: results/scenarios)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files already exist",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available scenarios and exit",
    )

    args = parser.parse_args()

    # List scenarios and exit
    if args.list:
        print("\nAvailable scenarios:")
        print("=" * 80)
        for scenario_id, name in SCENARIO_NAMES.items():
            print(f"  {scenario_id:2d}: {name}")
        print("=" * 80)
        return

    # Determine which scenarios to download
    if args.all:
        scenario_ids = list(range(1, 11))
    else:
        scenario_ids = args.scenario_id
        # Validate scenario IDs
        invalid_ids = [sid for sid in scenario_ids if sid not in range(1, 11)]
        if invalid_ids:
            parser.error(
                f"Invalid scenario IDs: {invalid_ids}. Must be between 1 and 10.")

    # Validate years
    valid_years = [2030, 2035, 2040]
    invalid_years = [y for y in args.years if y not in valid_years]
    if invalid_years:
        parser.error(
            f"Invalid years: {invalid_years}. Must be one of {valid_years}.")

    print("\n" + "=" * 80)
    print("🚀 E-Fuels Supply Potentials - Network File Downloader")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Scenarios: {scenario_ids}")
    print(f"  Years: {args.years}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Output dir: {args.output_dir or 'results/scenarios (default)'}")
    print(f"  Skip existing: {not args.force}")

    # Download files
    results = download_scenario_networks(
        scenario_ids=scenario_ids,
        years=args.years,
        resolution=args.resolution,
        output_dir=args.output_dir,
        skip_existing=not args.force
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
