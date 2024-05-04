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

The complete `Fuzzy System` was then benchmarked against some test data to evaluate its performance. (See the `./rule-benchmark.ipynb` notebook for more details.)

The resulting data for the `Fuzzy System` can be found in the `./fuzzy-inputs.txt`, `./fuzzy-outputs.txt`, and `./fuzzy-rules.txt` files.

### Approach 1

The first approach is created by myself by using all the splits of the `Decision Tree` for a given dimension to create some `Fuzzy Rules`. (I created it before I found the paper mentioned in `Approach 2`... In hindsight, I should have searched for a paper first...)

1. Collect All Splits for a given dimension.
   1. Each dimension consists of a set of splits `[-inf, split_1, split_2, ..., split_n, inf]`.
   2. Those splits form regions.
   3. For each region insert a `Fuzzy Set` in this case I used centered Gaussian for internal regions and sigmoid for the outer regions.
2. Translate the Decision Tree splits "Rule" formats (e.g., `if x > 0.5 and y < 2 then ...`)
   1. We see that in this rule the decision tree wants to split for all `x` values greater than `0.5`
   2. Select all previously created `Fuzzy Sets` that have their `center` greater than `0.5` and combine them with an `OR` operator.
        + Note: if the `smaller` comparison would be shorter. The whole calculation gets flipped using De Morgan's laws. This makes the rules smaller
   3. This should then activate for all `x` values greater than `0.5`.
   4. Continue this process for all splits in the dimension.

All the output variables are simply spread uniformly across a domain using `Gaussian` functions.

The resulting `Fuzzy System` is in this style:

```text
Input (subset):

Axis: "avgParticlesPerCell" range: (0.00, 8.16)
 "Very Low": Sigmoid(0.00100000, -1.47000006)
 "Low": Gaussian(0.13350000, 0.13250000)
 "Medium": Gaussian(0.44550000, 0.17950000)
 "High": Gaussian(3.13150000, 2.50650000)
 "Very High": Sigmoid(5.63800000, 1.47000006)

Rules (subset):

if (("avgParticlesPerCell" != "Very Low")) && (("avgParticlesPerCell" != "Low")) && (("avgParticlesPerCell" == "Very High")) && (("maxParticlesPerCell" == "Very High")) then ("Traversal" == "lc_sliced_c02")
if (("avgParticlesPerCell" == "Very Low") || ("avgParticlesPerCell" == "Low")) && (("numParticles" != "Very Low")) && (("numParticles" != "Low")) && (("maxParticlesPerCell" != "High")) && (("maxParticlesPerCell" != "Very High")) then ("Traversal" == "vcl_c06")
if (("avgParticlesPerCell" != "Very Low")) && (("avgParticlesPerCell" != "Low")) && (("avgParticlesPerCell" != "Very High")) && (("maxParticlesPerCell" == "Very Low")) && (("numParticles" == "High") || ("numParticles" == "Very High")) && (("numParticles" != "Very High")) then ("Traversal" == "vlc_sliced_c02")
...

Output (subset):

Axis: "Container" range: (0.00, 3.00)
 "LinkedCells": Gaussian(0.60000000, 0.15000000)
 "VerletClusterLists": Gaussian(1.20000000, 0.15000000)
 "VerletLists": Gaussian(1.80000000, 0.15000000)
 "VerletListsCells": Gaussian(2.40000000, 0.15000000)
```

### Approach 2

The second approach follows the paper [On constructing a fuzzy inference framework using crisp decision trees](https://www.sciencedirect.com/science/article/pii/S0165011406002533).

Its main idea is to directly convert the `Decision Tree` into a `Fuzzy Decision Tree`.

1. The main drawback of normal `Decision Trees` is that they make hard cuts at each stage.
2. The paper suggests using a `Linguistic` Variable` for each node of the decision tree. That fuzzily decides if the sample should proceed to the left or right child.
3. After transforming the `Decision Tree` in this way it's very easy to create a `Fuzzy System`. One just needs to extract all `Fuzzy Sets` and traverse the tree to create the rules.

The output variables are again spread uniformly across a domain using `Gaussian` functions.

The resulting `Fuzzy System` is in this style:

```text
Input (subset):

Axis: "avgParticlesPerCell" range: (-3.45, 3.99)
 "lower than 0.266": SigmoidFinite(3.98538651, 0.26600000, -3.45338651)
 "higher than 0.266": SigmoidFinite(-3.45338651, 0.26600000, 3.98538651)
 "lower than 5.638": SigmoidFinite(9.35738651, 5.63800000, 1.91861349)
 "higher than 5.638": SigmoidFinite(1.91861349, 5.63800000, 9.35738651)
 "lower than 0.001": SigmoidFinite(3.72038651, 0.00100000, -3.71838651)
 "higher than 0.001": SigmoidFinite(-3.71838651, 0.00100000, 3.72038651)
 "lower than 0.625": SigmoidFinite(4.34438651, 0.62500000, -3.09438651)
 "higher than 0.625": SigmoidFinite(-3.09438651, 0.62500000, 4.34438651)
 "lower than 0.002": SigmoidFinite(3.72138651, 0.00200000, -3.71738651)
 "higher than 0.002": SigmoidFinite(-3.71738651, 0.00200000, 3.72138651)
 "lower than 0.166": SigmoidFinite(3.88538651, 0.16600000, -3.55338651)
 "higher than 0.166": SigmoidFinite(-3.55338651, 0.16600000, 3.88538651)

Rules (subset):
if (("numEmptyCells" == "higher than 206.0")) && (("avgParticlesPerCell" == "lower than 0.266")) then ("Data Layout" == "SoA")
if (("numEmptyCells" == "lower than 206.0")) then ("Data Layout" == "SoA")
if (("avgParticlesPerCell" == "higher than 0.266")) && (("avgParticlesPerCell" == "lower than 5.638")) && (("maxParticlesPerCell" == "lower than 6.5")) && (("numParticles" == "lower than 20888.0")) && (("avgParticlesPerCell" == "lower than 0.625")) then ("Traversal" == "vlc_c18")

Output (subset):

Axis: "Container" range: (0.00, 3.00)
 "LinkedCells": Gaussian(0.60000000, 0.15000000)
 "VerletClusterLists": Gaussian(1.20000000, 0.15000000)
 "VerletLists": Gaussian(1.80000000, 0.15000000)
 "VerletListsCells": Gaussian(2.40000000, 0.15000000)

```

## Comparison

The rules created by `Approach 1` are more human-readable and can be easily understood. The rules created by `Approach 2` follow the idea of Fuzzy Decision Trees and are difficult to interpret directly. However, the size of the rules created by `Approach 1` quickly explodes and the rules become very large. This is not the case for `Approach 2` as those rules are limited by the depth of the decision tree.

Performance-wise both approaches perform similarly. The testing process is done in the `./rule-benchmark.ipynb` notebook.
