# Collected Tuning Data from WSL2

This is a collection of `LiveInfoStructs` and the corresponding `Configurations` (ranked by their performance). To keep the measured configurations accurate, $\Delta T = 0$ was used during the evaluations.

All the Configurations were purely evaluated by the `Full Tuning` method, so they should be optimally suited as training data for the `Fuzzy Tuning` method.

## Input File

The Input file that was used to generate this data can be found in the `./input.yaml` file.

All available CubeSpawners were used to generate the data. The parameters for the spawners were picked somewhat randomly.
The randomness should help to hopefully get a good spread of data which should cover a wide range of scenarios.

## Data collection Algorithm

1. Update the `input.yaml` file with (a) random 'Spawners' and (b) random 'Spawner Parameters'.
2. Simulate the new `input.yaml` file with `AutoPas`.
3. Save `LiveInfoStructs` from the simulation at iteration $0$.
4. Let the `Full Tuning` method run until it has tested all configurations.
5. Save the `Configurations` and their ranking by runtime at iteration $N$.
6. Stop the simulation and repeat the process from step 1.

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

## Results

The rules were created by using a `Decision Tree` algorithm by first finding good splits for the input data and then converting the splits into `Fuzzy Rules`.

The complete `Fuzzy System` was then benchmarked against some test data to evaluate its performance. (See the `./rule-bechmark.ipynb` notebook for more details.)

The resulting data for the `Fuzzy System` can be found in the `./fuzzy-inputs.txt`, `./fuzzy-outputs.txt`, and `./fuzzy-rules.txt` files.
