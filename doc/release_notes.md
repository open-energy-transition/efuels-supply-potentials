<!--
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Features and Contributions

Please list contributions, add reference to PRs if present.

* Integrated **generate aviation scenario** and **rescale fraction in airports dataset by state demand** into the workflow: [PR #35](https://github.com/open-energy-transition/efuels-supply-potentials/pull/35)

* Enabled **custom data moving** within workflow (`custom_powerplants.csv` and `NorthAmerica.csv`): [PR #21](https://github.com/open-energy-transition/efuels-supply-potentials/pull/21)

* Enabled **retrieving precomputed output** of power model's rules (e.g. `download_osm_data`, `clean_osm_data`, `build_shapes`, `build_osm_network`, `base_network`, and `build_renewable_profiles`) within the workflow: [PR #20](https://github.com/open-energy-transition/efuels-supply-potentials/pull/20)

* Integrated **US-specific cost assumptions** from NREL ATB to workflow: [PR #14](https://github.com/open-energy-transition/efuels-supply-potentials/pull/14) and technology-data [PR #1](https://github.com/open-energy-transition/technology-data/pull/1)

* Prepared **merged airports** dataset: [PR #16](https://github.com/open-energy-transition/efuels-supply-potentials/pull/16)

* Analyzed statewise **passengers and fuel consumption data** for fuel demand disaggregation for airports: [PR #9](https://github.com/open-energy-transition/efuels-supply-potentials/pull/9) 

* Facilitated **PyPSA-Earth sector run**: PyPSA-Earth [PR #1134](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1134), [PR #1143](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1143), [PR #1145](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1145), [PR #1165](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1165), [PR #1166](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1166)

* Fixed **hydro profile data** by splitting inflow to powerplants: PyPSA-Earth [PR #1119](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1119) 

* Added **statewise validation** script that compares demand and installed capacities with EIA data: [PR #7](https://github.com/open-energy-transition/efuels-supply-potentials/pull/7)

* Enabled import of pypsa-earth rules for **snakemake workflow management**: [PR #4](https://github.com/open-energy-transition/efuels-supply-potentials/pull/4), PyPSA-Earth [PR #1137](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1137) and [PR #1178](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1178)

* Added **country-level validation** script that compares the pypsa results with EIA and Ember data: [PR #1](https://github.com/open-energy-transition/efuels-supply-potentials/pull/1) and [PR #2](https://github.com/open-energy-transition/efuels-supply-potentials/pull/2)
