# Collected Tuning Data WSL2 Pause during Examples

This is a collection of `LiveInfoLogger` and `TuningDataLogger` results collected from all example scenarios of md-flexible.

All the Configurations were purely evaluated by the `Full Tuning` method, so they should be optimally suited as training data for the `Fuzzy Tuning` method.

To keep the measured configurations accurate, $\Delta T = 0$ was used **during** the tuning phases, such that all configurations are evaluated under the same conditions. Afterwards $\Delta T$ was set to its original value. This allows for different, accurate measurements of the scenario over its runtime.

## Input Files

All currently present example scenarios of md-flexible were used to collect the data.

## Data collection Algorithm

+ Compile with `-DAUTOPAS_LOG_TUNINGDATA=ON -DAUTOPAS_LOG_LIVEINFO=ON -DAUTOPAS_MIN_LOG_LVL=TRACE -DMD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING=ON -DAUTOPAS_LOG_TUNINGRESULTS=ON`. In order to pause the simulation during tuning.

1. Launch a example scenario with the `Full Tuning` method.
2. Wait for the simulation to finish.
3. Collect the `LiveInfoLogger` and `TuningDataLogger` results.
4. Repeat for all example scenarios.

## System

The data was collected in a `WSL2` environment on a `Windows 11` machine with the following specifications:

```text
CPU: AMD Ryzen 5 3600 6-Core Processor
Basisgeschwindigkeit: 3,59 GHz
Sockets: 1
Kerne: 6
Logische Prozessoren: 12
Virtualisierung: Aktiviert
L1-Cache: 384 KB
L2-Cache: 3,0 MB
L3-Cache: 32,0 MB
```
