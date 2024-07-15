# Structure: Exploring Fuzzy Tuning Technique for Molecular Dynamics Simulations in AutoPas

1. Introduction
   + Whats the role of Molecular Dynamics
   + Why is it important to tune the parameters
   + What is AutoPas? What are the current methods in AutoPas
   + Goal: Create and explore a framework to allow Fuzzy-Logic based tuning of parameters
     + Test wheter it is a viable alternative to current methods
2. Theoretical Background
   + Simulation of Molecular Dynamics
      + Newtons Equations
      + Lennard-Jones Potential
   + Fuzzy Logic
      + (High level: Implement a interpratable Function that maps continuous input values to continuous output values, Interpolation Problem)
      + Mathematical Background Mamdani
        + Fuzzification (Membership Functions)
        + Rule Base
        + Defuzzification
        + Example
   + Automatic Parameter Tuning
      + (many algorithmic approaches, no single best algorithm, depend on the problem and current state of the system)
      + Tunable Parameters
        + *(List of parameters)
      + *Current Methods
      + Try to remove as many bad combinations as possible bevore full search
3. Implementation
   + (High Level: Create interpretable fuzzy-rules to describe connections between paramters, Use Simulation Data LiveInfoStruct to apply these rules)
   + Input format XML / FML
   + Description of FuzzyTuner implements TuningStrategy
      + gets fed with LiveInfoStruct
      + applies rules
      + Defuzzification, and mapping | multiple approaches:
        + Line Ranking / Higher Dimensional Configuration Embedding
        + Suitability Approach
        + Tune Parameters seperately and merge them
4. Proof of Concept
   + To demonstrate the system a simple Proof of Concept Rule-Set is used
   + Sources of Rules
     + Reverse Engineer Full Tuning results and their connections to the LiveInfoStruct
       + Maybe show a plot of some discovered connections
     + Expert Knowledge
5. Comparison and Evaluation
   + How well does it work, compare to other methods
   + Benchmarks
6. Future Work
   + Problem: How to come up with the rules
      + Genetic Algorithm, Cost Function: Time
   + Problem: Dependence of parameters
      + Problem space blows up when just adding more parameters. Maybe nesting Fuzzy Systems works
7. Conclusion
8. Appendix
   + Input File
   + Rule File

## Graphs

+ Make Diagram of the Fuzzy System
  + Input Variables
  + Membership Functions
  + Rule Base
  + Output Variables
  + Defuzzification
  + Example
  + Maybe make a diagram of the FuzzyTuner
  + (Reuse rule-debugger.ipynb)

+ Make Diagram of certain rule activations
  + Plot 3d decision surface, and in which region what configuration is selected
  + Just extend rule-debugger.ipynb

+ Make diagram of "filter" to show how the sequential tuning straegies iteratevly remove configurations

## Additional Notes

+ Connection AutoPas and LS1Marydin

+ Make sections about findings of data collection



why decision trees? similar to the way humans make decisions, easy to understand, easy to interpret, easy to debug, easy to visualize, easy to explain, easy to implement, easy to handle missing values, easy to handle outliers