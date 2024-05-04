# Collecting data from the LiveInfoStruct

This is a collection of LiveInfoStructs and the corresponding Configurations ranked by their performance.

All the Configurations were purely evaluated by the `Full Tuning` method, to collect data that can be used to select appropriate rules for the `Fuzzy Tuning` method.

## Input File

The Input file that was used to generate this data can be found in the `./input.yaml` file.

All CubeSpawners were used to generate the data. The parameters for the spawners were picked somewhat randomly, to hopefully get a good spread of data which should cover a wide range of scenarios.

## Problems

I think not all configurations are tested evenly/properly, as not all configurations appear evenly in the collected data.

## System

The data was collected on a system with the following specifications:

It was collected inside `WSL2` on a `Windows 11` machine.

CPU
 AMD Ryzen 5 3600 6-Core Processor

 Basisgeschwindigkeit: 3,59 GHz
 Sockets: 1
 Kerne: 6
 Logische Prozessoren: 12
 Virtualisierung: Aktiviert
 L1-Cache: 384 KB
 L2-Cache: 3,0 MB
 L3-Cache: 32,0 MB
