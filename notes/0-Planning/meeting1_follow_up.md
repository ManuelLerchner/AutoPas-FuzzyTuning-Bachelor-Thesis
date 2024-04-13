# Follow up Meeting 1

## Possible Fuzzy Tuning Approaches

1. **Line Ranking**
   - Rank configurations on a line based on a user defined `?aggressiveness?` value
   - Use a single fuzzy logic system to calculate a crisp `?aggressiveness?` value
   - Pick the configuration closest to the crisp value
   - Drawback:
     - Need to specify a metric to somehow rank all configurations on a single axis
     - Can be improved with a `partial order` of configurations. Similar to the `rule-based-tuning` approach

2. **Suitability Approach**
   - Create many fuzzy systems, each for a single (full) configuration (e.g. 1 system for exactly: `(ContainerType=Verlet, Data=AOS, Traversal=C08, Newton3d=ON,...)`)
   - Each system produces a crisp value for the `suitability` of this configuration
   - Pick the lets say `k=5` configurations with the highest suitability and test all of them
   - Drawback:
     - Need to specify many rules to get a dense configuration space
  
3. **Determine each parameter separately**
   - Create a fuzzy system for each tunable parameter
     - e.g. one for `?traversal-size?` which justs selects `(LC04, LC08,...)`
     - one for `?container-complexity?` which just selects `(Verlet, LinkedCell,...)`
     - ...
   - This makes ranking easier, because you can just focus on rankinh the parameters instead of the whole configuration space
   - Each system decides what concrete Configuration-value each parameter should have
   - Combine all the “winners” to form the next Configurations (Cartesian product of all suggested parameters, filter out invalid configurations, tune with all of them)
   - Problems:
     - There are impossible combinations of parameters (But maybe this is not a problem, because you could filter those out)
     - Configuration parameters depend on each other  

## Possible Input file

Very high level idea, just to get a feeling for what needs to be specified in the input file.
Eventually it needs a czstom grammar and a interpreter to read in the fuzzy rules similar to the rule-based tuning approach.

```ts
// EXAMPLE fuzzy-rules.frule File //
// Probably shouldn't be in javascript notation but in a custom format //
// this is just for a rough idea //

// 1. Define Input Membership Functions
const fnum_particles = new Input(LiveInfo.num_particles)

fnum_particles.add_linguistic_term("low", new Step(between=[100,140], transition=[1,0])
fnum_particles.add_linguistic_term("average", new Triangle(center=200, steepness=20))
fnum_particles.add_linguistic_term("high", new Step(between=[400,1000], transition=[0,1])

let some_abstract_calculation = LiveInfo.parameterA / LiveInfo.parameterB
const faverage_distance= new Input(some_abstract_calculation)

faverage_distance.add_linguistic_term("low", ...)
faverage_distance.add_linguistic_term("average", ...)
faverage_distance.add_linguistic_term("high", ...)

...

// 2. Define Output Membership Functions

const fcan_parallelize= new Output()

fcan_parallelize.add_linguistic_term("little", center=0 ...)
fcan_parallelize.add_linguistic_term("moderate", ...)
fcan_parallelize.add_linguistic_term("good", center=100 ...)

...

// 3. Create some Rules based on fuzzy variables

Rule: // general rule
 IF fnum_particles is "high" and faverage_distance is "high" THEN fcan_parallelize is "good"

Rule: // general rule
 IF fnum_particles is "high" and faverage_distance is "high" THEN fcan_parallelize is "good"


// 4. State how the crisp output should be turned into a Configuration. Closest one wins

```

The different approaches could look like this:

```ts
// **Line Ranking**

const faggressiveness = new Output()

faggressiveness.add_linguistic_term("low", ...)
faggressiveness.add_linguistic_term("medium", ...)
faggressiveness.add_linguistic_term("high", ...)

Rule: 
 IF fnum_particles is "high" and faverage_distance is "high" THEN faggressiveness is "high"
  
FuzzyTuningResult.add_output_option([LC, AOS, NWT3=ON, ...], 42) // somehow the configuration [LC, AOS, NWT3=ON, ...] is ranked with aggressiveness 42
FuzzyTuningResult.add_output_option([LC, SOA, NWT3=OFF, ...], 35) // somehow the configuration [LC, SOA, NWT3=OFF, ...] is ranked with aggressiveness 35

... (Rank many more configurations based on their aggressiveness)

// The system will then pick the configuration with closest aggressiveness value to the crisp value

//this needs many more output options to capture all configurations. And somehow they need to be ranked
```

```ts
// **Suitability Approach**

const suitability_LC_04_AOS_NWT3_ON_... = new Output()

suitability_LC_04_AOS_NWT3_ON_....add_linguistic_term("low", ...)
suitability_LC_04_AOS_NWT3_ON_....add_linguistic_term("medium", ...)
suitability_LC_04_AOS_NWT3_ON_....add_linguistic_term("high", ...)

Rule: 
 IF fnum_particles is "high" and faverage_distance is "high" THEN suitability_LC_04_AOS_NWT3_ON_... is "high"

suitability_LC_04_AOS_NWT3_ON_.interpret_as_suitability_for([LC04, AOS, NWT3, ...]) // connect the suitability variable with its specific configuration. 

... (Do this with many, many more configurations)

//The system will then pick the best configurations based on the highest suitability values
```

```ts
// **Determine each parameter separately Style**

Rule: // general rule
 IF fnum_particles is "high" and faverage_distance is "high" THEN fcan_parallelize is "good"

Rule: // general rule
 IF fnum_particles is "high" and faverage_distance is "high" THEN fcan_parallelize is "good"

// Maybe it would be cool to suggest that its in general a good idea to use a certain data layout in 
// some cases. However, this introduces the problems of invalid configurations and you would need to somehow merge different suggested configurations. 
fcan_parallelize.add_output_option([*, Configuration.SOA, *, *, ...], 100) // this should mean that the system prefers SOA if crisp value \approx 100 and does not care about the other parameters
fcan_parallelize.add_output_option([*, Configuration.AOS, *, *, ...], 0)

... (Create more rules, for different tunable parameters)

// The system would need to merge all the different suggestions / expert knowledge suggestions configurations. Maybe just with a cartesian product of all suggested parameters. Here one could filter out invalid configurations. Then the tuning would need to test all of those configurations. 
// It should hopefully be cheaper than the full search space. 
```
