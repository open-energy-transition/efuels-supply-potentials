<!--
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Features and Contributions

Please list contributions, add reference to PRs if present.

* Add **custom ammonia, ethanol, cement and steel industries** [PR #50](https://github.com/open-energy-transition/efuels-supply-potentials/pull/50)

* Add **scenario configs for 2030, 2035 and 2040** and add data for scenario definition [PR #51](https://github.com/open-energy-transition/efuels-supply-potentials/pull/51)

* Cherry-pick: [PR #1400](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1400): Add **US-specific demand growth rates and fuel shares** (Medium scenario): [PR #52](https://github.com/open-energy-transition/efuels-supply-potentials/pull/52)

* Update head of the submodule to `efuels-supply-potentials` branch of `open-energy-transition/pypsa-earth` and cherry-pick [PR #1372](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1372): Scale temporal loads based on temporal resolution; [PR #1381](https://github.com/pypsa-meets-earth/pypsa-earth/pull/1381): Remove space in rail transport oil and electricity carriers: [PR #46](https://github.com/open-energy-transition/efuels-supply-potentials/pull/46)

* Added functionality to **connect all e-kerosene buses with a single E-kerosene-main bus** to model 0 cost of transport and store: [PR #42](https://github.com/open-energy-transition/efuels-supply-potentials/pull/42)

* Added functionality to **set data center loads** to `demand_profiles.csv`: [PR #41](https://github.com/open-energy-transition/efuels-supply-potentials/pull/41)

* Enabled **setting demand projections** based on the **NREL EFS**: [PR #38](https://github.com/open-energy-transition/efuels-supply-potentials/pull/38) and [PR #40](https://github.com/open-energy-transition/efuels-supply-potentials/pull/40)

* Added functionality to separate **e-kerosene** demand and set **SAF mandate** by choosing blending rate: [PR #37](https://github.com/open-energy-transition/efuels-supply-potentials/pull/37) 

* Integrated **generate aviation scenario** and **rescale fraction in airports dataset by state demand** into the workflow: [PR #35](https://github.com/open-energy-transition/efuels-supply-potentials/pull/35)

* Integrated **demand redistribution** based on utility and balancing authority level demand [PR #34](https://github.com/open-energy-transition/efuels-supply-potentials/pull/34)

* Updated **custom_powerplants.csv** data with new entries for coal and ror from EIA [PR #29](https://github.com/open-energy-transition/efuels-supply-potentials/pull/29)

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
