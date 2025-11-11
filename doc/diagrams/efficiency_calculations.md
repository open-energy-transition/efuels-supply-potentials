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
|efficiency |$1/gas_{input,DRI}$|$1/2.78(t_{DRI}/MWh_{gas})$|ton of DRI produced per MWh of gas|
|efficiency2|$-ore_{input,DRI}/gas_{input,DRI}$|$-1.59/2.78(t_{ore}/MWh_{gas})$|ton of iron-ore needed per MWh of gas|
|efficiency3|$i_{CO2,gas}$|$0.198(tCO_2/MWh_{gas})$|ton of $\text{CO}_2$ released to atmosphere per MWh of gas|

### DRI CC
If for DRI all gas goes for production of DRI, in DRI CC gas is used for DRI production and carbon capture (CC). Let's define gas fraction using in DRI as $x_{DRI}$, while fraction of gas used in CC as $x_{CC}$. So, $x_{DRI}+x_{CC} = 1$.
|parameter  |equation|values|detail|
|-----------|--------|------|------|
|efficiency |$x_{DRI}/gas_{input,DRI}$|$x_{DRI}/2.78(t_{DRI}/MWh_{gas})$|$x_{DRI}$ ton of DRI produced per MWh of gas|
|efficiency2|$-ore_{input,DRI}/gas_{input,DRI}$|$-1.59x_{DRI}/2.78(t_{ore}/MWh_{gas})$|$x_{DRI}$ ton of iron-ore needed per MWh of gas|
|efficiency3|$i_{CO2,gas}*(1-c_{steel,CC})$|$0.198*0.1(tCO_2/MWh_{gas})$|ton of $\text{CO}_2$ released to atmosphere per MWh of gas|
|efficiency4|$i_{CO2,gas}*c_{steel,CC}$|$0.198*0.9(tCO_{2,cap}/MWh_{gas})$|ton of $\text{CO}_2$ captured per MWh of gas|
|efficiency5|$-i_{CO2,gas}*c_{steel,CC}*elec-input_{DRI}$|$-0.198 * 0.9 * 0.16(MWh_{el}/MWh_{gas})$|electricity use in $MWh_{el}/MWh_{gas}$|

where $c_{steel,CC}$ - carbon capture rate for steel industry, $i_{CO2,gas}$ - carbon intensity of gas.

Fraction related to CC $x_{CC}$ can be found using gas-input for steel CC and captured amount of carbon:

$x_{CC}=gas_{input, steel-CC}*(c_{steel,CC}*i_{CO2,gas})$

$x_{CC} = 0.76 MWh_{gas}/tCO2_{cap} * 0.9 tCO2_{cap}/tCO2 * 0.198 tCO2/MWh_{gas} = 0.135432$

So, $x_{DRI} = 1-0.135432 = 0.864568$

Thus, we can use the efficiencies provided in the table above.

## BF-BOF and BF-BOF CC
Here is the information about the buses of BF-BOF industry.

|bus  |BF-BOF        |BF-BOF CC     |
|-----|--------------|--------------|
|bus0 |coal          |coal          |
|bus1 |steel BF-BOF  |steel BF-BOF  |
|bus2 |iron ore      |iron ore      |
|bus3 |co2 atmosphere|co2 atmosphere|
|bus4 |-             |co2 captured  |
|bus5 |-             |gas           |

### BF-BOF CC
The following equation describes the $CO_2$ capture per $MWh_{coal}$:

$CO_{2,cap} = (i_{CO2, coal} + k * i_{CO2, gas})*c_{steel,CC} $

where $k$ is a ratio of gas to coal energy usage with unit of $MWh_{gas}/MWh_{coal}$. We can call $k$ as **gas use rate** as compared with coal.

Gas use rate can be calculated alternatively using captured $CO_2$ and $gas_{input,CC}$:

$k (\text{gas use rate}) = (i_{CO2, coal} + k * i_{CO2, gas})*c_{steel,CC} * gas_{input,CC}$

From this eqation, we can obtain $k$ defined by other parameters:

$k = i_{CO2, coal} *c_{steel,CC} * gas_{input,CC} / (1 - i_{CO2, gas} * c_{steel,CC} * gas_{input,CC}) $

Gas use rate $k$ is directly used as `efficiency5`. While carbon capture efficiency (`efficiency4`) is defined as $(i_{CO2, coal} + k * i_{CO2, gas}) *c_{steel,CC}$

|parameter |equation|
|----------|--------|
|efficiency3|$(i_{CO2, coal} + k * i_{CO2, gas}) * (1 - c_{steel,CC})$|
|efficiency4|$(i_{CO2, coal} + k * i_{CO2, gas}) *c_{steel,CC}$|
|efficiency5|$-k$|

## Cement dry clinker and cement dry clinker CC

Here is the information about the buses of cement clinker industry.

|bus  |cement dry clinker|cement dry clinker CC|
|-----|--------------|--------------|
|bus0 |gas           |gas           |
|bus1 |clinker       |clinker       |
|bus2 |electricity   |electricity   |
|bus3 |co2 atmosphere|co2 atmosphere|
|bus4 |-             |co2 captured  |
|bus5 |-             |-             |

For cement dry clinker CC we have similar calculations as for DRI CC:

|parameter  |equation|values|detail|
|-----------|--------|------|------|
|efficiency |$x_{clinker}/gas_{input,clinker}$|$x_{clinker}/0.0002(t_{clinker}/MWh_{gas})$|$x_{clinker}$ ton of dry clinker produced per MWh of gas|
|efficiency2|$-(elec_{clinker}+elec_{CC})$|- |electricity use in $MWh_{el}/MWh_{gas}$|
|efficiency3|$i_{CO2,gas}*(1-c_{steel,CC})$|$0.198*0.1(tCO_2/MWh_{gas})$|ton of $\text{CO}_2$ released to atmosphere per MWh of gas|
|efficiency4|$i_{CO2,gas}*c_{steel,CC}$|$0.198*0.9(tCO_{2,cap}/MWh_{gas})$|ton of $\text{CO}_2$ captured per MWh of gas|

where $elec_{clinker}$ is electricity use by clinker process and equal to:

$elec_{clinker} = elec_{input,clinker}/gas_{input,clinker} $

$elec_{CC}$ is electricity use by carbon capture:

$elec_{CC} = i_{CO2,gas}*c_{steel,CC} * elec_{input,cc}$

And, $x_{clinker} =x_{DRI} = 1-0.135432 = 0.864568$