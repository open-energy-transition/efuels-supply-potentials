# SPDX-FileCopyrightText: Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Download solved PyPSA network files for scenarios from Google Drive.

Usage examples:
    # Download base year network
    python download_scenario_networks.py --base-year

    # Download all files for scenario 2
    python download_scenario_networks.py --scenario-id 2

    # Download all files for scenarios 1, 3, and 5
    python download_scenario_networks.py --scenario-id 1 3 5

    # Download all scenarios
    python download_scenario_networks.py --all

    # Download base year and scenarios
    python download_scenario_networks.py --base-year --scenario-id 1 2

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

# Base year network (2023 validation)
BASE_YEAR_URL = (
    "https://drive.google.com/file/d/1yhudhbgk_lTq7dJ4noDVxTXl37jhiCE2/view?usp=sharing"
)
BASE_YEAR_FILENAME = (
    "elec_s_100_ec_lcopt_Co2L-{resolution}_{resolution}_2020_0.07_AB_0export.nc"
)

# Default links for 3H scenario runs.
# Format: {scenario_id: {year: google_drive_url}}
NETWORK_FILES = {
    1: {
        # https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
        2030: "https://drive.google.com/file/d/1zQBZpGzpmws1LgEhDUeIIGkhbR9xZeEh/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1AR8uf5ILUb7LrDlFTaw8IlPoeVyeCHHa/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1u-50-QYZgY5Eslop4nVLvOIYyi_YhKVb/view?usp=drive_link",
    },
    2: {
        2030: "https://drive.google.com/file/d/1N9MvbmbLOu0xEiqQdvCSFu_H1VzNFOlm/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/13GBHDlYPITrKRnNYVd50k8LJkHf25zfa/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1tS9kpPFS4Kp40lyPjAYmgnHd_iIFLfMh/view?usp=drive_link",
    },
    3: {
        2030: "https://drive.google.com/file/d/1M6IbNgHHGhBKquMCzuNU5h-g4ohwdnM2/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/16dZrj0KLnO3CmgP-Xif8EWlwkA4H75Zz/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1XF_zGSkNZS8j455qptU5DxK7B1XkGp7k/view?usp=drive_link",
    },
    4: {
        2030: "https://drive.google.com/file/d/1PML9hpvXdou7F40nWyPfCkuLzk_7tAfw/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1sTXi68t-mMPP7Sr61ycTiwkGbqGFR8ik/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1-gurWDNlGdChNjUkmmh2FYtfdqx7QImJ/view?usp=drive_link",
    },
    5: {
        2030: "https://drive.google.com/file/d/1oSxQOqxatasUrCfqgD6SRBuVBddcCMYW/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1pqvYvAy0fWRAGFduN9a1m3rqtEHnN9Yp/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1w2clUUeRN4AUAuNYkc7D72iEAvw0X03E/view?usp=drive_link",
    },
    6: {
        2030: "https://drive.google.com/file/d/1aeXARxAJf3wYNgy8PWrRpN2-7S-Bn9T_/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1h6EexrKMQzsC8sh2D9uD9J-HwqAzLARo/view?usp=drive_link",
        2040: "PLACEHOLDER_URL_S06_2040",
    },
    7: {
        2030: "https://drive.google.com/file/d/1l-N4x_TWdHjbvK6w6aN3_7Pedu5LWUkb/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1MajNHWi8cJV4wv4A39OOeXbVoCxvfput/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1rXSaJHu5ua3_jlaoeI9iqXh6aWVU6tR-/view?usp=drive_link",
    },
    8: {
        2030: "https://drive.google.com/file/d/1OVkxarbe9rXW07mSTF92TSLZi0EblSKa/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1ASE4gEYzPgnRAX-G_qyJTlbrSGp_CS60/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1AWDrsev6XL0asBKINxD3-puz26bGE6iA/view?usp=drive_link",
    },
    9: {
        2030: "https://drive.google.com/file/d/1c6n1EVYrSIG_m-Vm3JBMk3jP5doaOfXN/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/16n-FQa5IKtP9Z8A-MgZMA6E9OSbwaVqK/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/1-IyZamUKQ_5dr2h72mmkPTJdXLf6iCaT/view?usp=drive_link",
    },
    10: {
        2030: "https://drive.google.com/file/d/1jBwL70UItnaa41S_rrgsr24ENBFukkSH/view?usp=drive_link",
        2035: "https://drive.google.com/file/d/1G3sjmHljLG-1LIPisb29gbByDMijgrYP/view?usp=drive_link",
        2040: "https://drive.google.com/file/d/18TJn6Vcq2Ct8kPdJZ4JhS2W-IlqOh1vQ/view?usp=drive_link",
    },
}

# Resolution-specific links for runs that differ from default 3H outputs.
# Format: {resolution: {scenario_id: {year: google_drive_url}}}
NETWORK_FILES_BY_RESOLUTION = {
    "1H": {
        2: {
            2030: "https://drive.google.com/file/d/1nxwLDCORUUuCJ3GwWNer0x0r8egDjDZI/view?usp=drive_link",
            2035: "https://drive.google.com/file/d/13tG8jZTVq2JQ35hCq_Jx6MpGDBFLLyB_/view?usp=drive_link",
            2040: "PLACEHOLDER_URL_S02_1H_2040",
        }
    }
}

# Network file naming patterns
# Scenarios 5, 6 use 'lcopt' wildcard; others use 'lv1'
FILENAME_PATTERNS = {
    1: "elec_s_100_ec_lv1_CCL-3H_3H_{year}_0.07_AB_0export.nc",
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


def download_file_from_google_drive(
    file_id: str, destination: Path, filename: str
) -> bool:
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
            file_id=file_id, dest_path=str(dest_path), unzip=False, showsize=True
        )

        # Verify the download was successful
        if dest_path.exists():
            file_size = dest_path.stat().st_size
            # Check if file is suspiciously small (likely an error page)
            if file_size < 10 * 1024 * 1024:  # Less than 10MB
                with open(dest_path, "rb") as f:
                    header = f.read(1024)
                    if (
                        b"<html" in header.lower()
                        or b"<!doctype html" in header.lower()
                    ):
                        print(
                            f"      ❌ Downloaded file is HTML error page ({file_size / 1024:.1f} KB)"
                        )
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


def download_base_year(
    resolution: str = "3H",
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
) -> bool:
    """
    Download base year network file.

    Args:
        resolution: Temporal resolution (default: 3H)
        output_dir: Base output directory (default: results/base_year)
        skip_existing: Skip download if file already exists

    Returns:
        True if download successful, False otherwise
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "base_year"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"📦 Base Year Network (2023)")
    print(f"{'=' * 80}")

    # Generate filename
    filename = BASE_YEAR_FILENAME.format(resolution=resolution)
    destination = output_dir / filename

    # Check if file already exists
    if skip_existing and destination.exists():
        file_size = destination.stat().st_size / (1024**3)  # Size in GB
        print(f"  ✓ Already exists ({file_size:.2f} GB) - Skipping")
        return True

    # Check for placeholder
    if "PLACEHOLDER" in BASE_YEAR_URL:
        print(f"  ⚠️  No Google Drive URL configured (placeholder found)")
        print(f"      Please update BASE_YEAR_URL in the script")
        return False

    # Extract file ID
    file_id = extract_file_id_from_url(BASE_YEAR_URL)
    if not file_id:
        print(f"  ❌ Invalid Google Drive URL format")
        return False

    # Download file
    print(f"  ⬇️  Downloading {filename}...")
    success = download_file_from_google_drive(file_id, output_dir, filename)

    if success:
        file_size = destination.stat().st_size / (1024**3)  # Size in GB
        print(f"  ✓ Download complete ({file_size:.2f} GB)")
        return True
    else:
        print(f"  ❌ Download failed")
        return False


def get_google_drive_url(scenario_id: int, year: int, resolution: str) -> Optional[str]:
    """Return URL for scenario/year/resolution, if configured."""
    resolution_links = NETWORK_FILES_BY_RESOLUTION.get(resolution, {})

    if scenario_id in resolution_links and year in resolution_links[scenario_id]:
        return resolution_links[scenario_id][year]

    # Use default links only for default 3H downloads.
    if resolution == "3H":
        return NETWORK_FILES.get(scenario_id, {}).get(year)

    return None


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
                f"⚠️  Warning: Scenario {scenario_id} not found in configuration. Skipping."
            )
            continue

        print(f"\n{'=' * 80}")
        print(f"📦 Scenario {scenario_id:02d}: {SCENARIO_NAMES[scenario_id]}")
        print(f"{'=' * 80}")

        scenario_dir = output_dir / f"scenario_{scenario_id:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        results[scenario_id] = {}

        for year in years:
            if year not in NETWORK_FILES[scenario_id]:
                print(
                    f"  ⚠️  Year {year} not available for scenario {scenario_id}. Skipping."
                )
                results[scenario_id][year] = False
                continue

            # Generate filename
            filename = FILENAME_PATTERNS[scenario_id].format(year=year)
            if resolution != "3H":
                # Replace resolution in filename if different from default
                filename = filename.replace("3H_3H", f"{resolution}_{resolution}")

            destination = scenario_dir / filename

            # Check if file already exists
            if skip_existing and destination.exists():
                file_size = destination.stat().st_size / (1024**3)  # Size in GB
                print(f"  ✓ {year}: Already exists ({file_size:.2f} GB) - Skipping")
                results[scenario_id][year] = True
                continue

            # Get Google Drive URL for requested resolution
            gdrive_url = get_google_drive_url(scenario_id, year, resolution)

            if not gdrive_url:
                print(
                    f"  ⚠️  {year}: No URL configured for scenario {scenario_id} at {resolution}"
                )
                print(
                    "      Please add links in NETWORK_FILES_BY_RESOLUTION before downloading"
                )
                results[scenario_id][year] = False
                continue

            # Check for placeholder
            if "PLACEHOLDER" in gdrive_url:
                print(
                    f"  ⚠️  {year}: No Google Drive URL configured (placeholder found)"
                )
                print(
                    "      Please update the URL in NETWORK_FILES or NETWORK_FILES_BY_RESOLUTION"
                )
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
            success = download_file_from_google_drive(file_id, scenario_dir, filename)

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
    print(f"\n{'=' * 80}")
    print("📊 Download Summary")
    print(f"{'=' * 80}")

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
            f"{status} Scenario {scenario_id:02d}: {scenario_success}/{scenario_total} files downloaded"
        )

    print(f"\n{'=' * 80}")
    print(f"Total: {successful}/{total_files} successful, {failed} failed")
    print(f"{'=' * 80}")


# -----------------------------
# CLI Interface
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download solved PyPSA network files from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download base year network
  python download_scenario_networks.py --base-year

  # Download all years for scenario 2
  python download_scenario_networks.py --scenario-id 2

  # Download multiple scenarios
  python download_scenario_networks.py --scenario-id 1 3 5 10

  # Download all scenarios
  python download_scenario_networks.py --all

  # Download base year and scenarios
  python download_scenario_networks.py --base-year --scenario-id 1 2

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

    parser.add_argument(
        "--base-year",
        action="store_true",
        help="Download base year (2023) network",
    )

    group = parser.add_mutually_exclusive_group(required=False)
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
        print("  Base year: 2023 validation network")
        print()
        for scenario_id, name in SCENARIO_NAMES.items():
            print(f"  {scenario_id:2d}: {name}")
        print("=" * 80)
        return

    # Check if at least base-year or scenario-id is provided
    if not args.base_year and not args.scenario_id and not args.all:
        parser.error(
            "At least one of --base-year, --scenario-id, or --all must be specified"
        )

    # Determine which scenarios to download
    scenario_ids = []
    if args.all:
        scenario_ids = list(range(1, 11))
    elif args.scenario_id:
        scenario_ids = args.scenario_id
        # Validate scenario IDs
        invalid_ids = [sid for sid in scenario_ids if sid not in range(1, 11)]
        if invalid_ids:
            parser.error(
                f"Invalid scenario IDs: {invalid_ids}. Must be between 1 and 10."
            )

    # Validate years
    valid_years = [2030, 2035, 2040]
    invalid_years = [y for y in args.years if y not in valid_years]
    if invalid_years:
        parser.error(f"Invalid years: {invalid_years}. Must be one of {valid_years}.")

    print("\n" + "=" * 80)
    print("🚀 E-Fuels Supply Potentials - Network File Downloader")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base year: {args.base_year}")
    print(f"  Scenarios: {scenario_ids if scenario_ids else 'None'}")
    print(f"  Years: {args.years}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Output dir: {args.output_dir or 'results/ (default)'}")
    print(f"  Skip existing: {not args.force}")

    # Download base year if requested
    base_year_success = None
    if args.base_year:
        base_year_success = download_base_year(
            resolution=args.resolution,
            output_dir=args.output_dir / "base_year" if args.output_dir else None,
            skip_existing=not args.force,
        )

    # Download scenario files
    results = {}
    if scenario_ids:
        results = download_scenario_networks(
            scenario_ids=scenario_ids,
            years=args.years,
            resolution=args.resolution,
            output_dir=args.output_dir / "scenarios" if args.output_dir else None,
            skip_existing=not args.force,
        )

    # Print summary
    if base_year_success is not None or results:
        print(f"\n{'=' * 80}")
        print("📊 Download Summary")
        print(f"{'=' * 80}")

        if base_year_success is not None:
            status = "✓" if base_year_success else "❌"
            print(
                f"{status} Base year: {'Downloaded' if base_year_success else 'Failed'}"
            )

        if results:
            print()
            print_summary(results)
        else:
            print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
