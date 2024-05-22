#!/usr/bin/env python
# coding: utf-8

# In[158]:

if True:
    import sys
    import os
    # ../../../notes/1-Testing/fuzzy-test/python/')
    currentdir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(1, os.path.join(currentdir, '../demo/fuzzy-test/python/'))

from fuzzy_system import LinguisticVariable, CrispSet, SigmoidFinite, make_continuousSet
from fuzzy_system import LinguisticVariable, CrispSet, Set
import seaborn as sns
from fuzzy_system import Sigmoid, Gaussian
import os
import sys
import numpy as np
import re
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import itertools

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


# In[170]:


def scatter_2d(df, x_name, y_name, label_name, filter=None):
    fig = plt.figure()
    fig.suptitle(f"2D Scatter Plot: {x_name} vs {y_name} for {label_name}")
    ax = fig.add_subplot(111)

    x = df[x_name]
    y = df[y_name]
    labels = df[label_name]

    markers = ['o', 'x', 's', 'v', '^']
    for i, c in enumerate(np.unique(labels)):
        if filter is None or c in filter:
            ax.scatter(x[labels == c], y[labels == c],
                       label=c, marker=markers[i % len(markers)])

    for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        t.set_rotation(45)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.legend()


def scatter_3d(df, x_name, y_name, z_name, label_name, filter=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(f"3D Scatter Plot: {x_name} vs {
                 y_name} vs {z_name} for {label_name}")

    x = df[x_name]
    y = df[y_name]
    z = df[z_name]
    labels = df[label_name]

    markers = ['o', 'x', 's', 'v', '^']
    for i, c in enumerate(np.unique(labels)):
        if filter is None or c in filter:
            ax.scatter(x[labels == c], y[labels == c], z[labels == c],
                       label=c, marker=markers[i % len(markers)])

    for t in [*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_zticklabels()]:
        t.set_rotation(45)

    # format ticklabes
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-4, 2))

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.legend()


# # Automatic Rule Learning
#

# In[171]:


class Condition:
    def __init__(self, feature, operator, value, negated=False):
        self.feature = feature
        self.operator = operator
        self.value = value
        self.negated = negated

    def split_features(self):

        if type(self.value) == str:
            values = self.value.split(",")

            if (len(values) > 1):
                pass

            conditions = map(lambda v: Condition(
                self.feature, self.operator, v.strip(), False), values)
            return list(conditions)
        return [self]

    def __str__(self):
        def format_value(x): return '"' + x + '"' if isinstance(x, str) else x

        return f"{
            '!' if self.negated else ''}({format_value(self.feature)} {self.operator} {format_value(self.value)})"


class Rule:
    # Condions are combined with AND, while the elements in a condition are combined with OR
    def __init__(self, conditions: list[list[Condition]], prediction):
        self.conditions = conditions
        self.prediction = prediction

    def __str__(self):
        # predictions = ""
        # for c in self.prediction.split_features():
        #     predictions += str(c) + " || "
        # # remove last " || "
        # predictions = predictions[:-4]

        # if self.prediction.negated:
        #     predictions = f"!({predictions})"

        predictions = str(self.prediction)

        return f"if {' && '.join(['(' + ' || '.join([str(c) for c in cond]) + ')' for cond in self.conditions])} then {predictions}"


# # Decision Tree
#

# In[172]:


def train_decision_tree(df, inputs, outputs, *args, **kwargs):
    labels = df[outputs]
    encoder = preprocessing.LabelEncoder()
    labels_enc = encoder.fit_transform(labels)

    ccp_alpha = kwargs.get("ccp_alpha", 0.0)

    # scale cc_alpha to the number of classes
    if ccp_alpha > 0:
        ccp_alpha = ccp_alpha / (np.log(3+len(np.unique(labels_enc))))
        kwargs["ccp_alpha"] = ccp_alpha

    model = DecisionTreeClassifier(random_state=1, **kwargs)

    model.fit(df[inputs], labels_enc, *args)

    score = model.score(df[inputs], labels_enc)

    return (model, encoder), score


def extract_rules_from_decision_tree(tree, feature_names, label_name, class_names) -> list[Rule]:
    tree_ = tree.tree_

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != -2:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1.append(Condition(name, "<=", np.round(threshold, 3)))
            recurse(tree_.children_left[node], p1, paths)
            p2.append(Condition(name, ">", np.round(threshold, 3)))
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        conditions = list(map(lambda x: [x], path[:-1]))

        classes = path[-1][0][0]
        l = np.argmax(classes)
        result = class_names[l]

        rules.append(Rule(conditions, Condition(label_name, "==", result)))

    return rules


def get_split_dimensions(tree, feature_names):
    tree_ = tree.tree_
    return set([feature_names[i] for i in tree_.feature if i != -2])


# # Helper Functions to learn and plot Decision Trees
#

# In[173]:


def find_rulesNd(df, inputs, label_name, *args, **kwargs):
    (model, pre), score = train_decision_tree(
        df, inputs, label_name, *args, **kwargs)

    rules = extract_rules_from_decision_tree(
        model, inputs, label_name, pre.classes_)

    return (model, pre), inputs, rules, score


markers = ['o', 'x', 's', 'v', '^']


def plut_rules_1d(df, model, encoder, inputs, used_inputs, label_name, score):
    (x_name,) = used_inputs
    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(16)
    fig.suptitle(f"{label_name}: {x_name} (score={score:.2f})")

    min_x = np.inf
    max_x = -np.inf

    for i, c in enumerate(np.unique(encoder.classes_)):

        ax[0].scatter(df[x_name][df[label_name] == c], np.zeros_like(df[x_name][df[label_name] == c]),
                      label=c, marker=markers[i % len(markers)])

        min_x = min(min_x, df[x_name][df[label_name] == c].min())
        max_x = max(max_x, df[x_name][df[label_name] == c].max())

    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100),
                         np.linspace(-1, 1, 100))

    data = {
        x_name: xx.ravel(),
    }

    for col in inputs:
        if col not in used_inputs:
            data[col] = np.zeros_like(xx.ravel())

    df_pred = pd.DataFrame(data)[inputs]

    Z = model.predict(df_pred)
    Z = Z.reshape(xx.shape)

    ax[0].contourf(xx, yy, Z, alpha=0.4)

    ax[0].set_xlabel(x_name)
    # disable y
    ax[0].get_yaxis().set_visible(False)

    ax[0].legend(prop={'size': 6})

    plot_tree(model, ax=ax[1], feature_names=inputs,
              class_names=encoder.classes_)


def plot_rules_2d(df, model, encoder, inputs, used_inputs, label_name, score):
    (x_name, y_name) = used_inputs

    fig, ax = plt.subplots(1, 2)
    fig.set_figwidth(16)
    fig.suptitle(f"{label_name}: {x_name} vs {y_name} (score={score:.2f})")

    min_x = np.inf
    max_x = -np.inf

    min_y = np.inf
    max_y = -np.inf

    for i, c in enumerate(np.unique(encoder.classes_)):

        ax[0].scatter(df[x_name][df[label_name] == c], df[y_name][df[label_name] == c],
                      label=c, marker=markers[i % len(markers)])

        min_x = min(min_x, df[x_name][df[label_name] == c].min())
        max_x = max(max_x, df[x_name][df[label_name] == c].max())

        min_y = min(min_y, df[y_name][df[label_name] == c].min())
        max_y = max(max_y, df[y_name][df[label_name] == c].max())

    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100),
                         np.linspace(min_y, max_y, 100))

    data = {
        x_name: xx.ravel(),
        y_name: yy.ravel()
    }

    for col in inputs:
        if col not in used_inputs:
            data[col] = np.zeros_like(xx.ravel())

    df_pred = pd.DataFrame(data)[inputs]

    Z = model.predict(df_pred)
    Z = Z.reshape(xx.shape)

    ax[0].contourf(xx, yy, Z, alpha=0.4)

    ax[0].set_xlabel(x_name)
    ax[0].set_ylabel(y_name)
    ax[0].legend(prop={'size': 6})

    plot_tree(model, ax=ax[1], feature_names=inputs,
              class_names=encoder.classes_)


def plot_rules_3d(df, model, encoder, inputs, used_inputs, label_name, score):
    (x_name, y_name, z_name) = used_inputs

    fig = plt.figure()
    fig.set_figwidth(16)
    fig.suptitle(f"{label_name}: {x_name} vs {
                 y_name} vs {z_name} (score={score:.2f})")

    ax = fig.add_subplot(121, projection='3d')

    for i, c in enumerate(np.unique(encoder.classes_)):
        ax.scatter(df[x_name][df[label_name] == c], df[y_name][df[label_name] == c], df[z_name][df[label_name] == c],
                   label=c, marker=markers[i % len(markers)])

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)

    # legend size small
    ax.legend(prop={'size': 6})

    ax = fig.add_subplot(122)

    plot_tree(model, ax=ax,  feature_names=inputs,
              class_names=encoder.classes_)


def plot_rules_Nd(df, model, encoder, inputs, used_inputs, label, score):
    fig = plt.figure()
    fig.set_figwidth(16)
    fig.suptitle(f"{label}: (score={score:.2f})")

    ax = fig.add_subplot(111)

    plot_tree(model, ax=ax,  feature_names=inputs,
              class_names=encoder.classes_)


# In[175]:

def create_auto_rules(X_train, y_train, weights, POSSIBLE_NUMBER_OF_COMBINATIONS, CCP_ALPHA, MAX_DEPTH, TOP_K_MODELS_PER_LABEL, exclude_columns=[]):
    df_filtered = pd.concat([X_train, y_train], axis=1)

    print(f"Training on {df_filtered.shape[0]} samples")

    tree_fits: dict[str, list[tuple[tuple[DecisionTreeClassifier,
                                          preprocessing.LabelEncoder], list[float], list[Rule], float]]] = {}

    for label in y_train.columns:
        for comb_size in POSSIBLE_NUMBER_OF_COMBINATIONS:

            cols = list(X_train.columns)
            for col in exclude_columns:
                cols.remove(col)

            for comb in itertools.combinations(cols, comb_size):

                tree_fit = find_rulesNd(
                    df_filtered, list(comb), label, weights, ccp_alpha=CCP_ALPHA[label], max_depth=MAX_DEPTH, class_weight='balanced')

                if label not in tree_fits:
                    tree_fits[label] = []

                tree_fits[label].append(tree_fit)

    auto_rules = {}

    for label in tree_fits:
        models = tree_fits[label]

        distinct_models = []
        seen_inputs = set()
        for ((model, encoder), inputs, rules, score) in models:
            used_inputs = frozenset(get_split_dimensions(model, inputs))

            if used_inputs not in seen_inputs:
                distinct_models.append(
                    ((model, encoder), inputs, rules, score))
                seen_inputs.add(used_inputs)

        # sort by score
        best_models = sorted(distinct_models, key=lambda x: x[3], reverse=True)[
            :TOP_K_MODELS_PER_LABEL]

        if label not in auto_rules:
            auto_rules[label] = []

        # plot the best models
        for (model, encoder), inputs, rules, score in best_models:

            used_inputs = get_split_dimensions(model, inputs)

            if (len(used_inputs) == 1):
                plut_rules_1d(df_filtered, model, encoder,
                              inputs, used_inputs, label, score)
            elif (len(used_inputs) == 2):
                plot_rules_2d(df_filtered, model, encoder,
                              inputs, used_inputs, label, score)
            elif (len(used_inputs) == 3):
                plot_rules_3d(df_filtered, model, encoder,
                              inputs, used_inputs, label, score)
            else:
                plot_rules_Nd(df_filtered, model, encoder,
                              inputs, used_inputs, label, score)

            auto_rules[label].extend(rules)

    print(f"Number of rules {sum([len(rules)
          for rules in auto_rules.values()])}:")
    for label, rules in auto_rules.items():
        print(f"\t{label} ({len(rules)} rules)")

    return auto_rules

# # Create Plots for Membership Functions
#

# Each Interval between two boundaries of the Decision Tree becomes a Membership Function.
#

# In[176]:


def cleanDimensionsBoundaries(auto_rules):
    dimensionsBoundariesAll: dict[str, list[float]] = {}

    for (label, rules) in auto_rules.items():
        for rule in rules:
            for andCond in rule.conditions:
                for condition in andCond:
                    feature = condition.feature
                    value = condition.value

                    if feature not in dimensionsBoundariesAll:
                        dimensionsBoundariesAll[feature] = [-np.inf, np.inf]
                    else:
                        dimensionsBoundariesAll[feature].append(value)

    # sort and remove duplicates
    for feature in dimensionsBoundariesAll:
        dimensionsBoundariesAll[feature] = sorted(
            list(set(dimensionsBoundariesAll[feature])))

    # Clean up boundaries, remove close values

    dimensionsBoundaries: dict[str, list[float]] = {}
    for feature in dimensionsBoundariesAll:
        boundaries = sorted(dimensionsBoundariesAll[feature])

        average_diff = np.mean(
            list(filter(lambda x: x != np.inf, np.diff(boundaries))))

        new_boundaries = [boundaries[0]]
        for i in range(1, len(boundaries)):
            diff = boundaries[i] - new_boundaries[-1]
            if diff < 0.2 * average_diff:
                continue
            new_boundaries.append(boundaries[i])

        dimensionsBoundaries[feature] = new_boundaries

    # Print out the cleaned
    for feature in dimensionsBoundaries:
        print(f"{feature}: {dimensionsBoundariesAll[feature]}")
        print(f"{feature}: {dimensionsBoundaries[feature]}")
        print()

    return dimensionsBoundaries


# # Create Activation Functions
#

# In[177]:


def plot_linguistic_variable_on_data(df, dimensionsBoundaries, linguistic_variable, dim):
    fig, ax1 = plt.subplots(1, 1)
    ax2 = ax1.twinx()

    fig.set_figwidth(10)

    ax1.hist(df[dim], bins=40, alpha=0.5, label="Data", align='mid')

    for rule_range in sorted(dimensionsBoundaries):
        ax1.axvline(x=rule_range, color='r', linestyle='--',
                    label=f"Boundary: {rule_range}")

    min_x = df[dim].min()
    max_x = df[dim].max()

    if type(min_x) == str:
        min_x = 0

    if type(max_x) == str:
        max_x = len(dimensionsBoundaries)-1

    inputs = np.linspace(min_x-0.05*(max_x-min_x),
                         max_x+0.05*(max_x-min_x), 1000)

    linguistic_variable.plot(ax2, n=800)

    ax1.set_xlabel(dim)
    ax1.set_ylabel("Frequency")
    ax1.legend(loc='upper right')

    ax2.set_ylabel("Activation")
    ax2.legend(loc='lower right')

    # set ranges
    ax1.set_xlim(min(inputs), max(inputs))
    # disable x ticks


# # Approach 1
#
# Is a self made approach to use the rules of the Decision Tree to create a Fuzzy System.
#

# In[178]:

def create_rules_approach1(X_train, auto_rules):
    dimensionsBoundaries = cleanDimensionsBoundaries(auto_rules)
    inputs_approach1: dict[str, LinguisticVariable] = {}

    linguisticDescriptions = ["Extremely Low", "Very Low", "Low",
                              "Medium", "High", "Very High", "Extremely High"]

    N = 2

    for dim in dimensionsBoundaries:
        std_deviation = X_train[dim].std()
        min_x = X_train[dim].min()-N*std_deviation
        max_x = X_train[dim].max()+N*std_deviation

        base_set = Set(Set.SetType.CONTINUOUS, (min_x, max_x))
        crisp_set = CrispSet({(dim, base_set)})
        inputs_approach1[dim] = LinguisticVariable(crisp_set)

        boundaries = sorted(dimensionsBoundaries[dim])

        already_chosen_names = set()

        for i, (curr_b, next_b) in enumerate(zip(boundaries, boundaries[1:])):
            percentage = (i+1) / len(boundaries)
            i = int(percentage * len(linguisticDescriptions))
            name = linguisticDescriptions[i]

            if name in already_chosen_names:
                name = name + " " + str(i)
            already_chosen_names.add(name)

            if (next_b-curr_b == np.inf):
                boundary = curr_b if curr_b != -np.inf else next_b

                neighbour_range = boundaries[i -
                                             1:i+1] if i > 0 else boundaries[i+1:i+3]

                std = (neighbour_range[1] - neighbour_range[0]) / 2 if len(
                    neighbour_range) == 2 else np.inf
                if np.abs(std) == np.inf:
                    std = (X_train[dim].max() - X_train[dim].min()) / 2

                flipped = -1 if curr_b == -np.inf else 1
                std *= flipped

                inputs_approach1[dim].add_linguistic_term(
                    Sigmoid(name, boundary, 6/std))
            else:
                mean = (curr_b + next_b) / 2
                std = (next_b - curr_b) / 2
                inputs_approach1[dim].add_linguistic_term(
                    Gaussian(name, mean, std))

        plot_linguistic_variable_on_data(X_train, dimensionsBoundaries[dim],
                                         inputs_approach1[dim], dim)

    fuzzy_rules_approach1 = []
    for label, rules in auto_rules.items():
        for rule in rules:
            new_conditions = []
            for and_cond in rule.conditions:
                or_condition = []
                for condition in and_cond:
                    feature = condition.feature
                    operator = condition.operator
                    value = condition.value

                    applicable_functionsR = list(filter(
                        lambda f: eval(f"f.peak() {operator} {value}"), inputs_approach1[feature].linguistic_terms.values()))

                    applicable_functionsL = list(filter(
                        lambda f: eval(f"{value} {operator} f.peak()"), inputs_approach1[feature].linguistic_terms.values()))

                    if len(applicable_functionsR) <= len(applicable_functionsL):
                        for f in applicable_functionsR:
                            or_condition.append(
                                Condition(feature, "==", f.linguistic_term))
                    else:
                        for f in applicable_functionsL:
                            new_conditions.append([
                                Condition(feature, "==", f.linguistic_term, True)])

                if len(or_condition) > 0:
                    new_conditions.append(or_condition)

            new_rule = Rule(new_conditions, rule.prediction)

            fuzzy_rules_approach1.append(new_rule)

    return inputs_approach1, fuzzy_rules_approach1


# # Aproach 2
#
# Follows https://www.sciencedirect.com/science/article/pii/S0165011406002533
#

# In[180]:

def create_rules_approach2(X_train, auto_rules):

    # Spread of the membership functions
    N = 2

    inputs_approach_2: dict[str, LinguisticVariable] = {}

    decision_boundaries: dict[str, set[float]] = {}

    for (label, rules) in auto_rules.items():
        for rule in rules:
            for andCond in rule.conditions:
                for condition in andCond:
                    feature = condition.feature
                    operator = condition.operator
                    value = condition.value

                    if feature not in decision_boundaries:
                        decision_boundaries[feature] = set()

                    decision_boundaries[feature].add(value)

                    std_deviation = X_train[feature].std()

                    dm = value-N*std_deviation
                    dn = value+N*std_deviation

                    if feature not in inputs_approach_2:
                        min_x = X_train[feature].min()-N*std_deviation
                        max_x = X_train[feature].max()+N*std_deviation
                        crisp_set = CrispSet(
                            {(feature, make_continuousSet((min_x, max_x)))})
                        inputs_approach_2[feature] = LinguisticVariable(
                            crisp_set)

                    # add the membership function to the linguistic variable
                    linguistic_variable = inputs_approach_2[feature]

                    low = SigmoidFinite(
                        f"lower than {value}", dn, (dm+dn)/2, dm)
                    high = SigmoidFinite(
                        f"higher than {value}", dm, (dm+dn)/2, dn)

                    high.based_on = []

                    linguistic_variable.add_linguistic_term(low)
                    linguistic_variable.add_linguistic_term(high)

    for dim in inputs_approach_2:
        boundaries = sorted(decision_boundaries[dim])

        plot_linguistic_variable_on_data(X_train, boundaries,
                                         inputs_approach_2[dim], dim)

    fuzzy_rules_approach2 = []
    for (label, rules) in auto_rules.items():
        for rule in rules:
            new_conditions = []

            for andCond in rule.conditions:
                or_condition = []
                for condition in andCond:
                    feature = condition.feature
                    operator = condition.operator
                    value = condition.value

                    if operator == "<=" or operator == "<":
                        or_condition.append(
                            Condition(feature, "==", f"lower than {value}"))
                    else:
                        or_condition.append(
                            Condition(feature, "==", f"higher than {value}"))

                new_conditions.append(or_condition)

            new_rule = Rule(new_conditions, rule.prediction)

            fuzzy_rules_approach2.append(new_rule)

    return inputs_approach_2, fuzzy_rules_approach2


# # Create Output Membership Functions
#

# In[182]:

def create_output_membership_functions(y_train):
    outputRangeMembershipFunctions: dict[str, LinguisticVariable] = {}

    for col in y_train.columns:

        values = sorted(y_train[col].unique())

        def flatMap(f, items):
            return [item for sublist in map(f, items) for item in sublist]

        split_values = flatMap(lambda e: e.split(","), values)
        split_values = sorted(
            list(set(map(lambda e: e.strip(), split_values))))

        ##
        split_values = values

        base_set = Set(Set.SetType.CONTINUOUS, (0, len(split_values)-1))
        crisp_set = CrispSet({(col, base_set)})

        outputRangeMembershipFunctions[col] = LinguisticVariable(crisp_set)

        for i, entry in enumerate(split_values):
            spacing = (len(split_values)-1)/(len(split_values)+1)
            mean = (i+1)*spacing
            std = (spacing/len(split_values))
            outputRangeMembershipFunctions[col].add_linguistic_term(
                Gaussian(entry, mean, std))

        plot_linguistic_variable_on_data(
            y_train, split_values, outputRangeMembershipFunctions[col], col)

    return outputRangeMembershipFunctions


# # Save all the data
#

# In[183]:

def save_linguistic_variables(variables, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for linguistic_var in variables.values():
            (name, cset) = next(iter(linguistic_var.crisp_set.dimensions))
            (rl, ru) = cset.values

            f.write(f"FuzzyVariable: domain: \"{name}\" range: ({rl}, {ru})\n")
            for func in sorted(linguistic_var.linguistic_terms.values(), key=lambda x: x.peak()):
                func.crisp_set = None
                f.write(f"\t{func}\n")
            f.write("\n")

    with open(filename, "r") as f:
        print(f.read())


# In[184]:

def save_fuzzy_rules(rules, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    rule_outputs: dict[str, Rule] = {}

    for rule in rules:
        if rule.prediction.feature not in rule_outputs:
            rule_outputs[rule.prediction.feature] = []

        rule_outputs[rule.prediction.feature].append(rule)

    with open(filename, "w") as f:
        for rules in rule_outputs.values():
            for rule in rules:
                if len(rule.conditions) == 0:
                    continue
                f.write(f"{rule}\n")
            f.write("\n")

    with open(filename, "r") as f:
        print(f.read())
