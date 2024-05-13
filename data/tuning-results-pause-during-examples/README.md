# Collected Tuning Data: Pause During Examples

This is a collection of `LiveInfoLogger` and `TuningData` files collected from the cluster. The simulation was frozen during the tuning phases by $\Delta T = 0$ using the `-DMD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING=ON` flag to keep the measured configurations accurate.

All the Configurations were purely evaluated by the `Full Tuning` method, so they should be optimally suited as training data for the `Fuzzy Tuning` method.

## Input Files

The Input files that were used to generate this data are based on the example inputs (e.g. `explodingLiquid.yaml` and `fallingDrop.yaml` ...) that are provided in the `AutoPas` repository. I changed the tuning interval such that the tuning phases are more frequent.

## Data collection Algorithm

1. Pick an example input file.
2. Simulate the new `input.yaml` file with `AutoPas` using the flags mentioned above.
3. Wait until completion.
    + The `MD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING` flag will automatically pause the simulation during the tuning phases and resume the simulation after the tuning phase is completed. This means that every tuning phase will be based on a different, but realistic, state of the simulation.
    + This means that the collected data should be way more "realistic" than the first approach of just picking random spawners.
4. Extract the `LogFiles` from the simulation and repeat the process from step 1.

The data was collected with the following flags:

`-DAUTOPAS_LOG_TUNINGDATA=ON -DAUTOPAS_LOG_LIVEINFO=ON -DAUTOPAS_MIN_LOG_LVL=TRACE -DMD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING=ON -DAUTOPAS_LOG_TUNINGRESULTS=ON`

## Pausing Example

This video shows the pausing of the simulation during the tuning phase. Whenever the simulation is paused, the tuning can happen undisturbed.

![Pausing Example](./pause_during_tuning.mp4)
