#!/usr/bin/env python3

import subprocess
import os
from time import sleep

SCRIPT_TEMPLATE = """#!/bin/bash

#SBATCH -J FuzzyTuning_{{YAML_FILENAME}}_{{NUM_THREADS}}Threads
#SBATCH -o ./%x.%j.%N.out
#SBATCH -D ./AutoPas/build/examples/md-flexible
#SBATCH --get-user-env
#SBATCH --clusters={{CLUSTER}}
#SBATCH --partition={{PARTITION}}
#SBATCH --mail-type=all
#SBATCH --mem=2000mb
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=manuel.lerchner@tum.de
#SBATCH --export=NONE
#SBATCH --time=06:00:00

sleep $((RANDOM % 120))
OMP_NUM_THREADS={{NUM_THREADS}} ./md-flexible --yaml-filename {{YAML_FILENAME}} --log-level info {{PARAMS}}
"""

CURRENTLY_FREE_SLOTS = 5


def ready_monitor_progress(cluster):
    while True:
        res = subprocess.check_output(
            ["sacct", "--cluster", cluster, "-X", "-u", "ge47wer2"])

        running = res.count(b"RUNNING")
        pending = res.count(b"PENDING")

        print("Running: ", running)
        print("Pending: ", pending)

        if running + pending < CURRENTLY_FREE_SLOTS:
            print("Free slots available!")
            return True

        print("No free slots available! Waiting...")

        sleep(60)


def launch_job(scenario_file, params, num_threads, cluster, partition):
    script = SCRIPT_TEMPLATE.replace("{{NUM_THREADS}}", str(num_threads))
    script = script.replace("{{CLUSTER}}", cluster)
    script = script.replace("{{PARTITION}}", partition)
    script = script.replace("{{YAML_FILENAME}}", scenario_file)
    script = script.replace("{{PARAMS}}", params)

    scenario_name = scenario_file.split(".")[0]
    filename = scenario_name+"_"+params+"_"+str(num_threads)+".sh"
    with open(filename, "w") as f:
        f.write(script)

    print("Launching job with ", num_threads, " threads")
    subprocess.call(["sbatch", filename])

    os.remove(filename)
    sleep(5)


def printUsefullCommands():
    expected_compilation_flags = "-DAUTOPAS_LOG_TUNINGDATA=ON -DAUTOPAS_LOG_LIVEINFO=ON -DAUTOPAS_MIN_LOG_LVL=TRACE -DMD_FLEXIBLE_PAUSE_SIMULATION_DURING_TUNING=ON -DAUTOPAS_LOG_TUNINGRESULTS=ON"

    input("Please make sure that the compilation flags are set to: " +
          expected_compilation_flags + "\n\nPress Enter to continue...")

    print("Useful commands:")
    print("squeue --cluster=serial")
    print("scancel --cluster=serial id")
    print("sacct --cluster=serial -X -u ge47wer2")
    print("")


if __name__ == "__main__":
    printUsefullCommands()

    threads = [1, 12, 24, 28]
    scenario = {
        "explodingLiquid.yaml":
            {
                "tuning_algorithm": {
                    "bayesian-Search": [""],
                    "bayesian-cluster-Search": [""],
                    "": [""],  # full search
                    "predictive-tuning": [""],
                    "rule-based-tuning": ["--rule-filename tuningRules.rule"],
                    "fuzzy-tuning":  ["--fuzzy-rule-filename fuzzyRulesComponents.frule", "--fuzzy-rule-filename fuzzyRulesSuitability.frule"],
                }
            }
    }

    for scenario_name, data in scenario.items():
        for num_threads in threads:

            cluster, partition = ("serial", "serial_std") if num_threads <= 28 else (
                "inter", "teramem_inter")

            for tuning_algorithm, params in data["tuning_algorithm"].items():
                params = data["tuning_algorithm"][tuning_algorithm]

                new_scenario_name = scenario_name.split(
                    ".")[0] + "-" + tuning_algorithm+".yaml"

                for param in params:

                    # guard against too many jobs
                    # ready_monitor_progress(cluster)

                    launch_job(new_scenario_name, param,
                               num_threads, cluster, partition)
