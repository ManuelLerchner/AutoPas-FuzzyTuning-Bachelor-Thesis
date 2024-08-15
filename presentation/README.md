# Potential Structure of the Slides

1. **Introduction** (3 minutes)
   - AutoPas
       - What is AutoPas?
       - Why is AutoPas significant in particle simulation?
         - Benefits of Tunable Parameters
         - Short Overview of Tunable Parameters
       - How do tuning strategies play a role in AutoPas?
         - Prune the search space
         - Potentially, many bad configurations
         - Importance of efficient tuning strategies

2. **Fuzzy Tuning Strategy** (5 minutes)
    - Benefits of Fuzzy Logic. Compared to Rule-Based Systems
      - Complex Models with relatively few rules
        - Interpolation Effect
        - (Show decision surface)
      - Fewer rules to maintain
      - Create human-readable rules
    - Recap of Fuzzy Logic
      - Fuzzy Sets
        - Fuzzy Logic Operations
      - Linguistic Variables
      - Fuzzy Logic Rules
      - Fuzzy Inference
        - Defuzzification
    - How to use Fuzzy Logic in AutoPas
        - Inputs are Live Info Data
        - What is the output?
          - Component Tuning Approach
            - One system per component
            - Independent tuning
            - Suggest configurations matching all predictions
          - Suitability Tuning Approach
            - Predict suitability for every configuration
            - Only evaluate configurations with high suitability

3. **Implementation** (short) (2 minutes)
    - Fuzzy Logic Framework
      - Library for arbitrary fuzzy systems
      - Recursive definition of fuzzy systems
    - Specification via Rule File
      - Human-readable format
      - Easy to maintain
    - OutputMapper
      - How to turn the real-valued output into a configuration
      - Component Tuning Approach: Different output, based on value. "Ideal location for the component"
      - Suitability Tuning Approach: Trivial. Each configuration has a dedicated system ==> no need for mapping (trivial mapping)

4. **Proof of Concept** (4 minutes)
    - Data-Driven Rule Extraction
      - Decision Trees
      - Conversion of Decision Trees to Fuzzy Systems
    - Fuzzy Systems for md flexible
      - Data Collection
      - Data Preprocessing
      - Component Tuning Approach
      - Suitability Tuning Approach

5. **Comparison and Evaluation** (3 minutes)
    - Exploding Liquid Benchmark (Included in Training Data)
    - Spinodal Decomposition MPI (Related to Training Data)
    - Further Analysis
      - Quality of Predictions During Tuning Phases
      - Optimal Suitability Threshold
      - Generalization of Rule Extraction Process

6. **Future Work** (2 minutes)
    - Dynamic Rule Generation
    - Improving Tuning Strategies
    - Simplification of the Fuzzy System to Decision Trees

7. **Conclusion** (1 minute)
    - Summary of Findings
    - Impact
    - Final Thoughts
