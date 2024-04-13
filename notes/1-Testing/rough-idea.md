# Implementation Steps

1. Create an "AST" for the fuzzy rules
   + Later implement ANTLR grammar for this

2. Create a way to specify the fuzzy system:
   + `AND` implementation, `OR` implementation, `NOT` implementation...
   + Defuzzification methods
   + <https://de.mathworks.com/help/fuzzy/working-from-the-command-line.html>

3. Implement the fuzzy system
   + Fuzzy Variables
     + Input: needs "name", "List(MembershipFunction)"
     + Output: needs "name", "List(MembershipFunction)"
   + MembershipFunction: needs "name", "function_type", "parameters"
   + FuzzySystem: needs "List(Input)", "List(Output)", "List(Rule)"
     + make sure they are compatible
   + Rule: needs "BoolExpression", "OutputStatement"
