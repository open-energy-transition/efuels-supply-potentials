| Carrier        | Component        | Credit Application Logic                                                                          |
|----------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| `nuclear`      | `Generator`      | Apply if `build_year < 2024`                                                                                      |
| `hydro`        | `StorageUnit`    | Apply if `2021 < build_year < 2027`                                                                               |
| `biomass`      | `Generator`      | Apply if `build_year ≥ 2025`                                                                                      |
| `geothermal`   | `Generator`      | Apply if `build_year ≥ 2025`                                                                                      |
| `solar`        | `Generator`      | `build_year == 2025`: 100% credit; `build_year == 2026`: 60%; 'build_year`== 2027`: 20%; 'build_year' >=2028`: 0% |
| `onwind`       | `Generator`      | Same as `solar`                                                                                                   |
| `offwind-ac`   | `Generator`      | Same as `solar`                                                                                                    
| `offwind-dc`   | `Generator`      | Same as `solar`                                                                                                   |
