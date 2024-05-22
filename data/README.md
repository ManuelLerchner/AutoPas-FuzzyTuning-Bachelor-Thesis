# Data Cluster

Each subdirectory contains the tuning results for the specific scenario using one of `[1,4,12,24,28]` threads. All SLURM jobs were created and launched with the `CoolMuc2_launch.py` script.

All simulations were repeated two times.

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

The code was compiled with `gcc 11.2.0`.

See also: [CoolMUC](https://doku.lrz.de/coolmuc-2-11484376.html)
