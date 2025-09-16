<!--
# SPDX-FileCopyrightText:  Open Energy Transition gGmbH
#
# SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Features and Contributions

Please list contributions, add reference to PRs if present.

* Apply **pre-OB3 tax credits** for solar, wind, electrolyzers, point-source CO2 and DAC for selected scenarios [PR #99](https://github.com/open-energy-transition/efuels-supply-potentials/pull/99)

* Enable **custom busmap** to have the same clustering for all horizons and scenarios [PR #96](https://github.com/open-energy-transition/efuels-supply-potentials/pull/96)

* Implement **maximum capacity constraint for geothermal** electricity generation plants [PR #95](https://github.com/open-energy-transition/efuels-supply-potentials/pull/95)

* Implement **production tax credits** removing tax credits in the base year and reviewing tax credits to renewables based on latest developments (One Big Beautiful Bill) [PR #91](https://github.com/open-energy-transition/efuels-supply-potentials/pull/91)

* Include biomass in RES and CES constraints-eligible sources [PR #90](https://github.com/open-energy-transition/efuels-supply-potentials/pull/90)

* Add **capital cost for hydro** storage units [PR #93](https://github.com/open-energy-transition/efuels-supply-potentials/pull/93)

* Adjust RES/CES constraints to facilitate easy **addition (removal) of biomass into (from) RES sources** [PR #90](https://github.com/open-energy-transition/efuels-supply-potentials/pull/90)

* Extend lifetime of nuclear power plants to 60 years [PR #94](https://github.com/open-energy-transition/efuels-supply-potentials/pull/94)

* **Base year**: Remove filtering for plants with DateOut >= 2023, remove PTC to existing wind, solar and biomass plants, add PTC for existing hydro and nuclear power plants, remove data center load [PR #85](https://github.com/open-energy-transition/efuels-supply-potentials/pull/85)

* Add **data center loads for sector model** [PR #81](https://github.com/open-energy-transition/efuels-supply-potentials/pull/81) and [PR #84](https://github.com/open-energy-transition/efuels-supply-potentials/pull/84)

* Prepare config files to define scenarios and review electrolyzer carriers [PR #76](https://github.com/open-energy-transition/efuels-supply-potentials/pull/76)

* Adjust `prepare_costs` in custom scripts (necessary to use universal currency conversion) [PR #75](https://github.com/open-energy-transition/efuels-supply-potentials/pull/75)

* Implement **hourly matching for hydrogen production** [PR #73](https://github.com/open-energy-transition/efuels-supply-potentials/pull/73)

* Enable **myopic optimization** [PR #66](https://github.com/open-energy-transition/efuels-supply-potentials/pull/66)

* Add custom rule to **apply state/country-wise constraints for clean/RES technologies** [PR #53](https://github.com/open-energy-transition/efuels-supply-potentials/pull/53)

* Update link to **retrieve renewable profiles (now generated via Earth cutout)** [PR #65](https://github.com/open-energy-transition/efuels-supply-potentials/pull/65)

* Update **missing DateOut for custom powerplants**, add **a script to fill missing DateOut**, and enable **powerplant filtering for future scenarios** [PR #61](https://github.com/open-energy-transition/efuels-supply-potentials/pull/61) with cherry-pick [PR #39](https://github.com/open-energy-transition/pypsa-earth/pull/39) to a working branch `efuels-supply-potentials`

* Separate **biogenic CO2 stored and grid H2** [PR #60](https://github.com/open-energy-transition/efuels-supply-potentials/pull/60)

* Update **head for pypsa-earth submodule** [PR #58](https://github.com/open-energy-transition/efuels-supply-potentials/pull/58) due to merge of [PR #34](https://github.com/open-energy-transition/pypsa-earth/pull/34) to a working branch `efuels-supply-potentials`

* Add **custom ammonia, ethanol, cement and steel industries** [PR #50](https://github.com/open-energy-transition/efuels-supply-potentials/pull/50)

* Add **dynamic blending rate selection** rule [PR #55](https://github.com/open-energy-transition/efuels-supply-potentials/pull/55) 

* Add custom rule to **fetch scenario-dependent growth rates** for the US [PR #57](https://github.com/open-energy-transition/efuels-supply-potentials/pull/51)

* Add scenario configs for 2030, 2035 and 2040 and add data for scenario definition [PR #51](https://github.com/open-energy-transition/efuels-supply-potentials/pull/51) 

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
