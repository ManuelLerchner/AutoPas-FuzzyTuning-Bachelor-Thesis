#!/usr/bin/env python
# coding: utf-8
if True:
    import sys
    import os
    # ../../../notes/1-Testing/fuzzy-test/python/')
    currentdir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(1, os.path.join(currentdir, '../../../notes/1-Testing/fuzzy-test/python/'))

from collections import defaultdict
from fuzzy_system import *
import os
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


# In[1]:


# In[12]:
# imports

# In[13]:


# In[14]:


FOLDERS = ["approach1", "approach2"]


# In[15]:


def parseVariables(VariableString):
    inputVariables = []

    for axis in VariableString.split("\n\n"):
        if not axis:
            continue

        # Axis: maxParticlesPerCell range: (0.00, 35.00)
        nameStr = re.search(r"FuzzyVariable: domain: \"(.*)\"", axis).group(1)
        rangeStr = re.search(r"range: \((.*), (.*)\)", axis)

        variable = LinguisticVariable(
            CrispSet({(nameStr, make_continuousSet((float(rangeStr.group(1)), float(rangeStr.group(2)))))}))

        for rule in axis.strip().split("\n")[1:]:
            name = re.search(r"^\s*\"(.*)\":", rule).group(1)
            func = re.search(r": (\w+)\(", rule).group(1)
            params = re.search(r"\((.+)\)", rule).group(1).split(", ")
            if func == "Gaussian":
                variable.add_linguistic_term(Gaussian(name,
                                                      float(params[0]), float(params[1])))
            elif func == "Sigmoid":
                variable.add_linguistic_term(Sigmoid(
                    name, float(params[0]), float(params[1])))
            elif func == "SigmoidFinite":
                variable.add_linguistic_term(SigmoidFinite(
                    name, float(params[0]), float(params[1]), float(params[2])))
            else:
                raise Exception("Unknown function: "+func)

        inputVariables.append(variable)

    return inputVariables


# In[16]:


def constructFuzzySet(variables, terms: list[list[tuple[str, str]]]):
    fuzzySet = None
    for and_list in terms:
        or_set = None
        for (term, comp, value) in and_list:
            a = None
            for i in variables:
                for (name, _) in i.crisp_set.dimensions:
                    if name == term:
                        a = i
                        break

            if a is None:
                raise ValueError(f"Variable {term} not found")

            curr_set = (a == value) if comp == "==" else ~(a == value)

            or_set = curr_set | or_set if or_set is not None else curr_set

        fuzzySet = fuzzySet & or_set if fuzzySet is not None else or_set
    return fuzzySet


def parseRules(rulesString: str):
    andLists = []
    for antecedent in rulesString.split("&&"):
        andString = re.search(r"\((.+)\)", antecedent).group(1)
        orStrings = andString.split("||")

        negateAnd = False
        if (andString[0] == '!'):
            negateAnd = True

        or_list = []
        for orString in orStrings:
            term = re.search(r"\(?\"(.*)\" == \"(.*)\"\)?", orString)

            op = "!=" if negateAnd else "=="

            or_list.append((term.group(1), op, term.group(2)))

        andLists.append(or_list)

    return andLists


# In[17]:


def createFuzzySystem(inputVariableString, outputVariableString, rulesString):
    inputVariables = parseVariables(inputVariableString)
    outputVariables = parseVariables(outputVariableString)

    variables = inputVariables + outputVariables

    rules = []
    for rule in rulesString.strip().split("\n"):
        if not rule:
            continue
        antecedentsString = re.search(r"if (.+) then", rule).group(1)
        consequentList = re.search(r"then (.+)", rule).group(1)

        antecedent = constructFuzzySet(
            variables, parseRules(antecedentsString))

        consequent = constructFuzzySet(variables, parseRules(consequentList))

        rules.append(FuzzyRule(antecedent, consequent))

    algo_rankings = {}

    for output in outputVariables:
        name = next(iter(output.crisp_set.dimensions))[0]

        for linguistic_term, membershipFunc in output.linguistic_terms.items():
            peak = membershipFunc.peak()
            if name not in algo_rankings:
                algo_rankings[name] = {}
            algo_rankings[name][linguistic_term] = peak

    print("Number of rules: ", len(rules))
    print("Number of input variables: ", len(inputVariables))
    print("Number of output variables: ", len(outputVariables))

    fuzzy_systems = {}

    for rule in rules:
        dims = rule.consequent.crisp_set.dimensions
        assert len(dims) == 1
        output_variable = next(iter(dims))[0]

        if output_variable not in fuzzy_systems:
            fuzzy_systems[output_variable] = FuzzySystem()
        fuzzy_systems[output_variable].add_rule(rule)

    return fuzzy_systems, algo_rankings


# In[19]:


def calc_accuracy(train, test, fisys: FuzzySystem, algo_rankings: dict[str, dict[str, float]], K, n):
    trainCorrect = 0
    trainWrong = 0

    testCorrect = 0
    testWrong = 0

    output_variable = fisys.consequent_name

    for row in train.iterrows():
        vals, preds = fisys.predictClosest(
            row[1], algo_rankings[output_variable], n=n)

        true = row[1][output_variable]

        for pred in preds:
            if pred in true:
                trainCorrect += 1
            else:
                trainWrong += 1

    for row in test.iterrows():
        vals, preds = fisys.predictClosest(
            row[1], algo_rankings[output_variable], n=n)

        true = row[1][output_variable]

        for pred in preds:
            if pred in true:
                testCorrect += 1
            else:
                testWrong += 1

    return trainCorrect/(trainCorrect+trainWrong), testCorrect/(testCorrect+testWrong)


# In[20]:


def plot_accuracy(train, test, folder, fuzzy_systems, algo_rankings, K, n):
    accuracies = {}

    for label, fisys in fuzzy_systems.items():
        print(f"\n{label}:")

        pctTrain, pctTest = calc_accuracy(train, test,
                                          fisys, algo_rankings, K, n)

        print(f"Train: {pctTrain}")
        print(f"Test: {pctTest}")

        accuracies[label] = (pctTest, pctTrain)

    # create pi chart

    fig, ax = plt.subplots(1, len(accuracies), figsize=(16, 5))

    fig.suptitle(f"Accuracy of the fuzzy system on the test set for {folder}")

    for i, (label, (test, train)) in enumerate(accuracies.items()):
        ax[i].set_title(label)
        ax[i].pie([test, 1-test], labels=["Correct",
                                          "Incorrect"], autopct='%1.1f%%', startangle=90, colors=['green', 'red'])

    plt.show()

    fig, ax = plt.subplots(1, len(accuracies), figsize=(16, 5))

    fig.suptitle(f"Accuracy of the fuzzy system on the train set for {folder}")

    for i, (label, (test, train)) in enumerate(accuracies.items()):
        ax[i].set_title(label)
        ax[i].pie([train, 1-train], labels=["Correct",
                                            "Incorrect"], autopct='%1.1f%%', startangle=90, colors=['darkgreen', 'darkred'])

    plt.show()


# In[21]:

def benchmark_rules(folder, train, test, K=1, n=100):

    with open(folder+'/fuzzy-inputs.txt') as f:
        inputVariableString = f.read()

    with open(folder+'/fuzzy-outputs.txt') as f:
        outputVariableString = f.read()

    with open(folder+'/fuzzy-rules.txt') as f:
        rulesString = f.read()

    fiss, algo_ranking = createFuzzySystem(
        inputVariableString, outputVariableString, rulesString)

    print(f"\n{folder}:")

    plot_accuracy(train, test, folder, fiss, algo_ranking, K, n)

    return fiss, algo_ranking
