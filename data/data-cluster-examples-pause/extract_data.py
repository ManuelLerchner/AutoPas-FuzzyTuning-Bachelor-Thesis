
from collections import defaultdict
import os
import re
import shutil


folder = "./new"

out_files: dict = {}


for filename in os.listdir(folder):
    # filter files ending in .out
    if filename.endswith(".out"):
        # parse filename
        # "FuzzyTuning_fallingDrop.yaml_1Threads.4066898.i23r05c03s10"

        find = re.search(r'(.*).yaml_(\d+)Threads', filename)
        try:
            scenario = find.group(1)
            threads = find.group(2)
        except:
            print(f"Error parsing filename {filename}")
            raise Exception("Error parsing filename")

        # read file
        with open(os.path.join(folder, filename), "r") as f:
            lines = f.readlines()

        # find line having a timestamp [2024-05-22 21:15:54.174] [AutoPasLog] [info] [AutoPasImpl.h:65] AutoPas Version: 2.0.0-a46cd3a
        for line in lines:
            if "AutoPas Version" in line:
                # extract start time hour:minute:second
                start = re.search(r'\[(.*?)\]', line).group(1)
                start = start.replace(" ", "_")
                start = start.replace(":", "_")
                start = start.split(".")[0]

                # store start time
                if start in out_files:
                    print(out_files[start])
                    (sc, th, f) = out_files[start]

                    if sc == scenario and th == threads:
                        print(f"Conflict: {scenario} {
                              threads} vs {sc} {th} at {start}")
                        raise Exception("Conflict")

                out_files[start] = (scenario, threads, filename)

                break


# count number of duplicate scenarios and threads
scenarios = {}
for key in out_files:
    (scenario, threads, _) = out_files[key]

    if scenario not in scenarios:
        scenarios[scenario] = {}

    if threads not in scenarios[scenario]:
        scenarios[scenario][threads] = 0

    scenarios[scenario][threads] += 1

num_repeats = 1
for scenario in scenarios:
    for threads in scenarios[scenario]:
        if scenarios[scenario][threads] > 1:
            if num_repeats == -1:
                num_repeats = scenarios[scenario][threads]
            elif num_repeats != scenarios[scenario][threads]:
                print(f"Scenario {scenario} with {threads} has {
                      scenarios[scenario][threads]} repeats. Expected {num_repeats}")
                # raise Exception("Different number of repeats")

# go through all files and copy the csv files matching to a folder with the scenario name
# AutoPas_liveInfoLogger_Rank0_2024-05-22_21-16-57.csv
# AutoPas_tuningData_Rank0_2024-05-22_21-17-15

same_time: dict = {}


for filename in os.listdir(folder):
    # filter files ending in .csv
    if filename.endswith(".csv"):
        # parse filename
        # AutoPas_liveInfoLogger_Rank0_2024-05-22_21-16-57.csv
        timestamp = re.search(
            r'Rank0_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename).group(1)

        # replace last two - with _
        replace = timestamp.rsplit("-", 2)
        timestamp = replace[0] + "_" + replace[1] + "_" + replace[2]

        if timestamp not in same_time:
            same_time[timestamp] = []

        same_time[timestamp].append(filename)


# make directories
for timestamp in out_files.keys():
    (scenario, _, _) = out_files[timestamp]
    for i in range(1, num_repeats + 1):
        print(f"CREATE {folder}/{scenario}_{i}")
        if not os.path.exists(f"{folder}/{scenario}_{i}"):
            os.makedirs(f"{folder}/{scenario}_{i}")


current_repeat: dict = defaultdict(lambda: 1)

for timestamp in out_files.keys():
    (scenario, threads, out_file) = out_files[timestamp]
    csv_files = same_time[timestamp]

    # copy files to scenario_repeat folder
    for file in [out_file] + csv_files:
        print(f"COPY {file} to {
              folder}/{scenario}_{current_repeat[(scenario, threads)]}")

        # move file
        shutil.copy(
            f"{folder}/{file}", f"{folder}/{scenario}_{current_repeat[(scenario, threads)]}")

    current_repeat[(scenario, threads)] += 1
