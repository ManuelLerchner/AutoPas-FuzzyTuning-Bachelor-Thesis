# Collected Tuning Data Cluster Pause during Examples

This is a collection of `LiveInfoLogger` and `TuningDataLogger` results collected from all example scenarios of md-flexible on the CoolMUC2 cluster.

All the Configurations were purely evaluated by the `Full Tuning` method, so they should be optimally suited as training data for the `Fuzzy Tuning` method.

To keep the measured configurations accurate, $\Delta T = 0$ was used **during** the tuning phases, such that all configurations are evaluated under the same conditions. Afterwards, $\Delta T$ was set to its original value. This allows for different, accurate measurements of the scenario over its runtime.

Each subdirectory contains the tuning results for the specific scenario using one of `[1,4,12,24,28]` threads. All SLURM jobs were created and launched with the `CoolMuc2_launch.py` script.

All simulations were repeated two times. The code was compiled with `gcc 11.2.0`.

## Input Files

All currently present example scenarios of md-flexible were used to collect the data.

## Data collection Algorithm

+ Compile with `-DAUTOPAS_LOG_TUNINGDATA=ON -DAUTOPAS_LOG_LIVEINFO=ON -DAUTOPAS_MIN_LOG_LVL=TRACE -DMD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING=ON -DAUTOPAS_LOG_TUNINGRESULTS=ON`. In order to pause the simulation during tuning.

1. Execute the `CoolMuc2_launch.py` script to launch the SLURM jobs for the example scenarios.
2. Wait for the simulation to finish.
3. Copy all `Logs` from the `CoolMUC2` cluster to the local machine.
4. Edit the `CoolMuc2_launch.py` script to add new Scenarios and repeat from step 1.

## System

The data was collected on `CoolMUC2` with the following specifications:

```text
Hardware
Number of nodes: 812
Cores per node: 28
Hyperthreads per core: 2
Core nominal frequency: 2.6 GHz
Memory (DDR4) per node: 64 GB (Bandwidth 120 GB/s - STREAM)
Bandwidth to interconnect per node: 13,64 GB/s (1 Link)
Bisection bandwidth of interconnect (per island): 3.5 TB/s
Latency of interconnect: 2.3 µs
Peak performance of system: 1400 TFlop/s

Infrastructure
Electric power of fully loaded system: 290 kVA
Percentage of waste heat to warm water: 97%
Inlet temperature range for water cooling: 30 … 50 °C
Temperature difference between outlet and inlet: 4 … 6 °C

Software (OS and development environment)
Operating system: SLES15 SP1 Linux
MPI: Intel MPI 2019, alternatively OpenMPI
Compilers: Intel icc, icpc, ifort 2019
Performance libraries: MKL, TBB, IPP
Tools for performance and correctness analysis: Intel Cluster Tools
```

See also: [CoolMUC](https://doku.lrz.de/coolmuc-2-11484376.html)
