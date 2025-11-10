## Efficiency calculations for industry with CCS

The efficiency calculations for steel and cement industries are not straighforward. It is because we have *gas-input* values which defined gas needs for capturing 1 $\text{tCO}_2$. At the same time, captured amount of $\text{CO}_2$ depends on gas requirement of the process.

Below, we provide equations for defining efficiencies of the links in PyPSA-Earth for cement and steel industries.

## DRI and DRI CC
Here is the information about the buses of DRI industry.

|bus  |DRI           |DRI CC        |
|-----|--------------|--------------|
|bus0 |gas           |gas           |
|bus1 |DRI           |DRI           |
|bus2 |iron ore      |iron ore      |
|bus3 |co2 atmosphere|co2 atmosphere|
|bus4 |-             |co2 captured  |
|bus5 |-             |electricity   |

### DRI 
|parameter  |equation|values|detail|
|-----------|--------|------|------|
|efficiency |$\frac{1}{\text{gas-input}_{DRI}}$|$\frac{1}{2.78}$ $\frac{\text{t}_{DRI}}{\text{MWh}_{gas}}$|ton of DRI produced per MWh of gas|
|efficiency2|$\frac{\text{ore-input}_{DRI}}{\text{gas-input}_{DRI}}$|$\frac{1.59}{2.78}\frac{\text{t}_{ore}}{\text{MWh}_{gas}}$|ton of iron-ore needed per MWh of gas|
|efficiency3|$\text{CO}_2\text{-intensity}_{gas}$|$0.198\frac{\text{tCO}_2}{\text{MWh}_{gas}}$|ton of $\text{CO}_2$ released to atmosphere per MWh of gas|

### DRI CC
If for DRI all gas goes for production of DRI, in DRI CC gas is used for DRI production and carbon capture (CC). Let's define gas fraction using in DRI as $x_{DRI}$, while fraction of gas used in CC as $x_{CC}$. So, $x_{DRI}+x_{CC} = 1$.
|parameter  |equation|values|detail|
|-----------|--------|------|------|
|efficiency |$\frac{x_{DRI}}{\text{gas-input}_{DRI}}$|$\frac{x_{DRI}}{2.78}\frac{\text{t}_{DRI}}{\text{MWh}_{gas}}$|$x_{DRI}$ ton of DRI produced per MWh of gas|
|efficiency2|$\frac{\text{ore-input}_{DRI}}{\text{gas-input}_{DRI}}$|$\frac{1.59x_{DRI}}{2.78}\frac{\text{t}_{ore}}{\text{MWh}_{gas}}$|$x_{DRI}$ ton of iron-ore needed per MWh of gas|
|efficiency3|$i_{CO2,gas}*(1-c_{steel,CC})$|$0.198*0.1\frac{\text{tCO}_2}{\text{MWh}_{gas}}$|ton of $\text{CO}_2$ released to atmosphere per MWh of gas|
|efficiency4|$i_{CO2,gas}*c_{steel,CC}$|$0.198*0.9\frac{\text{tCO}_2}{\text{MWh}_{gas}}$|ton of $\text{CO}_2$ captured per MWh of gas|
|efficiency5|$i_{CO2,gas}*c_{steel,CC}*\text{elec-input}_{DRI}$|$0.198*0.9*0.16\frac{\text{MWh}_{el}}{\text{MWh}_{gas}}$|ton of $\text{CO}_2$ captured per MWh of gas|

where $c_{steel,CC}$ - carbon capture rate for steel industry, $i_{CO2,gas}$ - carbon intensity of gas.