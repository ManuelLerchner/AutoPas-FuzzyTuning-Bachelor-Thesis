# Notes Meeting 10.04.2024

## Preliminary: Mentioned Paper

- Used `Tpar` (loop execution time), `LIB` (load imbalance) and `ΔTpar` / `ΔLIB` to calculate `ΔDLS` (A metric ranking their scheduling algorithms on a numerical line)
- They have 2 modes: Directly calculating the DLS value, or calculating a deltaDLS value
  - **Example: High** load imbalance and **long** execution time should lead to a **more aggresive** DLS (deltaDLS positive)
- They assign each loop scheduling algorithm some abstract `DLS` value.
  - I dont really understand this value. They specify it as  $\frac{Load Imbalance}{Overhead}$
- This assumes that one can rank all possible outputs on a single numerical line

## Implementation Idea

- Similar to **Rule Based Tuning**
  - Make use of `LiveInfo` struct to access current parameter
  - And implement complete fuzzy-logic system
  - One could implement a similar `rule-grammar` to specify fuzzy-rules:

    ```tsx
    // EXAMPLE fuzzy-rules.frule File //
    
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
    
    // 3. Create Rules based on fuzzy variables
    
    Rule:
     IF fnum_particles is "high" and faverage_distance is "high" THEN fcan_parallelize is "good"
    
    Rule:
     IF fnum_particles is "high" and faverage_distance is "high" THEN fcan_parallelize is "good"
    
    let helper_variable = (not fnum_particles is "low") or faverage_distance = "average"
    
    Rule:
     IF helper_variable 
    
    // 4. State how the crisp output should be turned into a Configuration. Closest one wins
    fcan_parallelize.add_output_option(Configuration.SOA, 100)
    fcan_parallelize.add_output_option(Configuration.AOS, 0)
    ```

  - An Fuzzy system then calculates this at the end of every tuning phase

## Future Problems

## 1. Possible tuning approaches

### Maybe a Fuzzy System for each tunable parameter?

- One for Container Type
- One for `AOS` vs `SOA`
- One for Traversal
- ….
- Each system decides what concrete Configuration-value each parameter should have. Then all the “winners” get combined into the next Configuration
- (Requires some metric for each parameter, if there is no rule for a parameter need to test all combinations)

### Maybe rank Configurations based on aggressiveness somehow?

- But how can you even specify “aggressiveness” for all configurations?
- How should that even relate to the `LiveData` provided?
- I think this doesnt work

## 2. Tuning the rules

- Automatic tuning would be cool. However it needs some kind of `Cost Function`
  - Is there some smart metric to apply a genetic algorithm?
  - Cost function  cant work well if the selected configurations is simply the “closest” one matching the fuzzy output. Since there is no direct change between the output value and the selected configuration.
- Other idea:
  - Calculate all (a subset) of configurations and map the best one (time-wise / energywise) back on the “output-ranges” of the fuzzy systems. Then this empirical value could be used to tune the weights/membership functions
  - Example: If 0.8 would be mapped to L08 traversal. And L08 Traversal was the best one in the samples. This could be used to “infer” 0.8 as the true-desired value for this case. Maybe one could use a softmax to give probabilities to those back-mappings

## 3. Generally: How to even come up with the ruleset?

- What is `expert-knowlege` in  `md-flexible` simulations?

## General Questions

- How are the meetings supposed to happen?
  - On a regular basis?
  - What should I prepare?
- How does a BA thesis work?
  - Do I just work on my own and keep you updated?
  - Do you have any concrete things you want to try?
  - Do I just fork the AutoPas repo and make a pull request?
    - What am I allowed to do?
    - Can i freely add a `Grammar` and `Parser`
    - Do I just dump everything in a `fuzzy-rule-based-tuning` folder?
    - Who is responsible for Code Review and merging?
- Communication?
  - Via Email or via the Matrix Chat?
  - How often?
  - About what?
- I need a “Proof that my bachelors thesis is in English” for my masters application
  - What counts as proof?
