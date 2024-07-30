# Bachelor's Thesis: "Exploring Fuzzy Tuning Technique for Molecular Dynamics Simulations in AutoPas"

This repository contains all the collected data, scripts, and the final thesis document for my `Bachelor Thesis` at the Technical University of Munich (TUM).
The thesis was written at the `Chair of Scientific Computing` under the supervision of [Manish Kumar Mishra](https://www.cs.cit.tum.de/en/sccs/personen/manish-kumar-mishra/) and [Samuel Newcome](https://www.cs.cit.tum.de/en/sccs/people/samuel-james-newcome).

The thesis explores a novel tuning technique for molecular dynamics simulations in the [AutoPas](https://github.com/AutoPas/AutoPas) framework. The technique is based on fuzzy logic and aims to predict close-to-optimal simulation configurations from a set of live data collected during the simulation. The goal of this thesis is to evaluate the feasibility of the fuzzy tuning technique and to compare it with existing tuning techniques.

## Quick Guide for Creating New Rule Sets

This section provides a quick guide on how to create new rule sets similar to the ones developed in this thesis. (
[Suitability.frule](https://github.com/AutoPas/AutoPas/blob/fuzzy-tuning-strategy/examples/md-flexible/input/fuzzyRulesSuitability.frule) and [Components.frule](https://github.com/AutoPas/AutoPas/blob/fuzzy-tuning-strategy/examples/md-flexible/input/fuzzyRulesComponents.frule))

> [!TIP]
>
> 1. **Collect Data (Logfiles of AutoPas Simulations)**:
>
>    + To enable logging make sure to enable: `-DAUTOPAS_LOG_TUNINGDATA=ON` `-DAUTOPAS_LOG_LIVEINFO=ON` `-DAUTOPAS_LOG_TUNINGRESULTS=ON` or similar in the CMake configuration.
>
>    + `-DMD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING=ON` is also recommended to guarantee fair comparisons between different configurations.
>
>    + The script `data/CoolMuc2_launch_DataCollection.py` provides a template for launching such simulations on the CoolMuc2 cluster.
>    + You should end up with a bunch of log files for each evaluated scenario consisting of: `AutoPas_tuningData_*.csv`, `AutoPas_tuningResults_*.csv`, `AutoPas_iterationPerformance_*.csv` and `Scenario_*.out` files. Copy them to your local machine and place them in the `data/new` folder.
>    + If you want to cleanup the `*.out` files, you can use the script `data/remove_empty_lines.sh` script and run it in the `data/new` directory.
>
> 2. **Sort Data into Folders**:
>
>    + Currently all logfiles are placed in the `data/new` directory, with only the timestamp as a hint for the actual scenario. To create meaningful datasets, we sort all logfiles of the same scenario into a separate folder. You can use the script `data/extract_data.py` to automatically attempt to sort the logfiles into folders based on the timestamps present in the `*.out` and `*.csv` files.
>    + Check all the created folders manually and verify that all logfiles are correctly placed in the corresponding scenario folder.
>
> 3. **Create Permanent Data Folders**:
>    + Copy the created folders in `data/new` to a permanent location in the `data` directory.
>
> 4. **Create Rule Sets**:
>    + Go to the `data-analysis` directory and adapt one of the templates `rule-extraction-components-template.ipynb` or `rule-extraction-suitability-template.ipynb` to your needs.
>       + You will need to adapt the PATH to the new data folders in the first cell of the notebook.
>    + After running the entire notebook, you will find the extracted rule sets in the `suitability-rules` or `components-rules` directory.
>
>    + Unfortunately, you need to manually create the output mappings in the `*.frule` files. You can look at [Suitability.frule](https://github.com/AutoPas/AutoPas/blob/fuzzy-tuning-strategy/examples/md-flexible/input/fuzzyRulesSuitability.frule) or [Components.frule](https://github.com/AutoPas/AutoPas/blob/fuzzy-tuning-strategy/examples/md-flexible/input/fuzzyRulesComponents.frule) for reference. Tools such as Github-Copilot can help with this repetitive task.
>
>    + The resulting `*.frule` files can now be used in AutoPas with the Fuzzy Tuning Strategy by adding `--fuzzy-rule-filename {RULES}.frule` to the command line arguments.
>
>    + If you want you can format the rules to your liking and make them more readable.

## Thesis

The thesis is available in LaTeX format in this repository. You can access the rendered version in PDF format by clicking the following link:

[Read the Thesis (PDF)](latex/Manuel_Lerchner_Fuzzy_Tuning.pdf)

### Table of Contents

TODO

### Abstract

TODO

## Slides

The presentation slides are available in PDF format and can be accessed via the following link:

[View the Slides (PDF)](presentation/slides.pdf)
