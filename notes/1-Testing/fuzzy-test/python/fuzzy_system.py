#!/usr/bin/env python
# coding: utf-8

# In[84]:


import matplotlib.pyplot as plt
import numpy as np
import functools


# In[85]:


from enum import Enum


class Set:
    class SetType(Enum):
        CONTINUOUS = 1
        DISCRETE = 2

    def __init__(self, type: SetType, values: any):
        self.type = type
        self.values = values

    def get_coverage(self, grid_size: int):
        if self.type == Set.SetType.CONTINUOUS:
            return np.linspace(self.values[0], self.values[1], grid_size)
        elif self.type == Set.SetType.DISCRETE:
            return self.values

    def __repr__(self):
        return f"Set({self.type}, {self.values})"


def make_continuousSet(range: tuple[float, float]):
    return Set(Set.SetType.CONTINUOUS, range)


def make_discrete(values: list[float]):
    return Set(Set.SetType.DISCRETE, values)


class CrispSet:
    def __init__(self, dimensions: set[tuple[str, Set]]):
        """
        A class representing a crisp set.

        dimensions: [(name, (min, max))] A list of tuples where the first element is the name of the dimension and the second element is a tuple representing the minimum and maximum value of the dimension.
        With the help of the cartesian product, the dimensions get combined.
        """
        self.dimensions = dimensions

    def __mul__(self, other):
        """
        The cartesian product of two crisp sets.
        """
        new_dimensions = set()
        new_dimensions.update(self.dimensions)
        new_dimensions.update(other.dimensions)

        return CrispSet(new_dimensions)

    def __repr__(self):
        return str(self.dimensions)


# In[86]:


class FuzzySet:
    def __init__(self, linguistic_term: str, function: callable, based_on: list = [], is_base_set=False, crisp_set: CrispSet = None) -> None:
        """
        A class to represent a fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        crisp_set: The crisp set that the fuzzy set is defined over.
        function: A function that takes a value and returns the membership value of that value in the fuzzy set.
        based_on: A list of fuzzy sets that this fuzzy set is based on. This is used for plotting purposes.
        """
        self.linguistic_term = linguistic_term
        self.crisp_set = crisp_set
        self.function = function
        self.based_on = based_on
        self.is_base_set = is_base_set

    def __call__(self, data):
        if self.is_base_set:
            # Function is a direct function of the input data
            try:
                if (len(self.crisp_set.dimensions) != 1):
                    raise ValueError(
                        "Can only calculate membership for one variable")
                (name, _) = next(iter(self.crisp_set.dimensions))
                return self.function(data[name])
            except:
                return self.function(data)
        else:
            # Function is a function a lambda function combining other fuzzy sets
            # Pass the data down recursively
            return self.function(data)

    def defuzzyfy(self, numpoints=100):
        return self.mom(numpoints)

    def cog(self, numpoints=100):
        """
        Calculate the center of gravity of the fuzzy set.

        Returns: (x,y) The x and y coordinates of the center of gravity.
        Uses the range of the crisp set to calculate the center of gravity numerically.
        """
        assert len(
            self.crisp_set.dimensions) == 1, "Can only calculate center of gravity for one variable"
        (name, cset) = next(iter(self.crisp_set.dimensions))

        numX = 0
        numY = 0
        den = 0

        for x in cset.get_coverage(numpoints):
            y = self.function({name: x})
            numX += x*y
            numY += 0.5*y*y
            den += y

        if den == 0:
            return (0, 0)

        return (numX/den, numY/den)

    def mom(self, numpoints=100):
        """
        Calculate the mean of maximum of the fuzzy set.

        Returns: (x,y) The x and y coordinates of the mean of maximum.
        Uses the range of the crisp set to calculate the mean of maximum numerically.
        """

        assert len(
            self.crisp_set.dimensions) == 1, "Can only calculate mean of maximum for one variable"
        (name, range) = next(iter(self.crisp_set.dimensions))

        max_y = 0

        places_of_max = []

        for x in range.get_coverage(numpoints):
            y = self.function({name: x})
            if y > max_y:
                max_y = y
                places_of_max = [x]
            elif y == max_y:
                places_of_max.append(x)

        if len(places_of_max) == 0:
            return (0, 0)

        mean = sum(places_of_max) / len(places_of_max)

        return (mean, max_y)

    def __and__(self, other):
        """
        The intersection of two fuzzy sets.
        """
        new_linguistic_term = f"({str(self)} and {str(other)})"
        new_crisp_set = self.crisp_set * other.crisp_set
        return FuzzySet(new_linguistic_term, lambda data: min(self(data), other(data)), [self, other], crisp_set=new_crisp_set)

    def __or__(self, other):
        """
        The union of two fuzzy sets.
        """
        new_linguistic_term = f"({str(self)} or {str(other)})"
        new_crisp_set = self.crisp_set * other.crisp_set
        return FuzzySet(new_linguistic_term, lambda data: max(self(data), other(data)), [self, other], crisp_set=new_crisp_set)

    def __invert__(self):
        """
        The negation of a fuzzy set.
        """
        new_linguistic_term = f"not {str(self)}"
        new_crisp_set = self.crisp_set
        return FuzzySet(new_linguistic_term, lambda data: 1 - self(data), [self], crisp_set=new_crisp_set)

    def plot(self, ax):
        assert len(
            self.crisp_set.dimensions) == 1, "Can only plot fuzzy sets over one variable"
        [name, set] = next(iter(self.crisp_set.dimensions))
        xrange = set.get_coverage(1000)

        for mf in self.based_on:
            [_, mf_set] = next(iter(mf.crisp_set.dimensions))
            yh = [mf({name: x}) for x in xrange]

            if mf_set.type == Set.SetType.DISCRETE:
                total_range = max(mf_set.values) - min(mf_set.values)
                ax.bar(xrange, yh, label=mf.linguistic_term,
                       width=total_range/(50*len(mf_set.values)), alpha=0.5)
            else:
                ax.plot(xrange, yh, label=mf.linguistic_term,
                        linestyle='--', alpha=0.5, linewidth=0.5)

        y = [self({name: x}) for x in xrange]

        if set.type == Set.SetType.DISCRETE:
            total_range = max(set.values) - min(set.values)
            ax.bar(xrange, y, label=str(self),
                   width=total_range/(50*len(set.values)))
        else:
            ax.plot(xrange, y, label=str(self))

        if len(self.based_on) > 0:
            if set.type == Set.SetType.CONTINUOUS:
                ax.fill_between(xrange, 0, y, alpha=0.25)
            (cog_x, cog_y) = self.defuzzyfy()
            ax.axvline(cog_x, color='black', linestyle='--')
            ax.plot([cog_x], [cog_y], marker='o', markersize=5, color="black",
                    label=f"Defuzzification: ({cog_x:.2f}, {cog_y:.2f})")

        ax.set_xlabel(name)
        ax.set_ylabel("Membership Value")

        ax.legend()

    def __repr__(self):
        return f"\"{self.linguistic_term}\""


class Triangle(FuzzySet):

    def __init__(self, linguistic_term, center, width):
        """
        A class representing a triangular fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        center: The center of the fuzzy set. (The point where the membership value is 1)
        width: The width of the triangular shape. (Half of the base of the triangle) (The points where the membership value is 0)
        """

        self.center = center
        self.width = width
        def function(x): return max(0, 1 - abs(x - center) / width)
        super().__init__(linguistic_term, function, is_base_set=True)

    def peak(self):
        return self.center

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Triangle({self.center:.8f}, {self.width:.8f})"


class Trapezoid(FuzzySet):

    def __init__(self, linguistic_term, left, center_left, center_right, right):
        """
        A class representing a trapezoidal fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        left: The left edge of the fuzzy set. (The point where the membership value is 0)
        center_left: The left center of the fuzzy set. (The point where the membership value is 1)
        center_right: The right center of the fuzzy set. (The point where the membership value is 1)
        right: The right edge of the fuzzy set. (The point where the membership value is 0)
        """

        self.left = left
        self.center_left = center_left
        self.center_right = center_right
        self.right = right

        if (self.left == self.center_left):
            self.left -= 0.001
        if (self.right == self.center_right):
            self.right += 0.001

        def function(x): return max(0, min((x - self.left) / (self.center_left -
                                                              self.left), 1, (self.right - x) / (self.right - self.center_right)))
        super().__init__(linguistic_term, function, is_base_set=True)

    def peak(self):
        return (self.center_left + self.center_right) / 2

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Trapezoid({self.left:.8f}, {self.center_left:.8f}, {self.center_right:.8f}, {self.right:.8f})"


class Gaussian(FuzzySet):

    def __init__(self, linguistic_term, mean, sigma):
        """
        A class representing a Gaussian fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        mean: The mean of the Gaussian. (The point where the membership value is 1)
        sigma: The standard deviation of the Gaussian. (The width of the Gaussian)
        """

        self.mean = mean
        self.sigma = sigma
        def function(x): return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
        super().__init__(linguistic_term, function, is_base_set=True)

    def peak(self):
        return self.mean

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = next(iter(self.crisp_set.dimensions))[0] + " is "
        return f"{prefix}\"{self.linguistic_term}\": Gaussian({self.mean:.8f}, {self.sigma:.8f})"


class Sigmoid(FuzzySet):

    def __init__(self, linguistic_term, center, width):
        """
        A class representing a sigmoid fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        center: The center of the sigmoid. (The point where the membership value is 0.5)
        width: The width of the sigmoid. (The steepness of the sigmoid)
        """

        self.center = center
        self.width = width
        def function(x): return 1 / (1 + np.exp(-width * (x - center)))
        super().__init__(linguistic_term, function, is_base_set=True)

    def peak(self):
        return np.inf if self.width > 0 else -np.inf

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Sigmoid({self.center:.8f}, {self.width:.8f})"


class SigmoidFinite(FuzzySet):
    def __init__(self, linguistic_term, dm, beta, dn):
        """
        A class representing a finite sigmoid fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        lower: The lower bound of the sigmoid. (The point where the membership value is 0)
        center: The center of the sigmoid. (The point where the membership value is 0.5)
        upper: The upper bound of the sigmoid. (The point where the membership value is 1)
        """

        self.dm = dm
        self.beta = beta
        self.dn = dn

        invert = self.dm > self.dn

        if invert:
            dm, dn = dn, dm

        def function(x):
            if x <= dm:
                return 0
            if dm <= x <= beta:
                return 0.5*((x-dm)/(beta-dm))**2
            if beta <= x <= dn:
                return 1-0.5*((x-dn)/(beta-dn))**2
            return 1

        if invert:
            def func(x): return 1 - function(x)
        else:
            def func(x): return function(x)

        super().__init__(linguistic_term, np.vectorize(func), is_base_set=True)

    def peak(self):
        return np.inf if self.dn > self.dm else -np.inf

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": SigmoidFinite({self.dm:.8f}, {self.beta:.8f}, {self.dn:.8f})"


class Singleton(FuzzySet):

    def __init__(self, linguistic_term, value):
        """
        A class representing a singleton fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        value: The value of the singleton fuzzy set. (The point where the membership value is 1)
        """

        self.value = value
        def function(x): return 1 if x == value else 0
        super().__init__(linguistic_term, function, is_base_set=True)

    def peak(self):
        return self.value

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Singleton({self.value:.8f})"


# In[88]:


class LinguisticVariable:
    def __init__(self, crisp_set: CrispSet) -> None:
        """
        A class to represent a fuzzy variable.

        crisp_set: The name of the crisp set. (e.g. "age")
        """
        self.crisp_set = crisp_set
        self.linguistic_terms: dict[str, FuzzySet] = dict()

    def add_linguistic_term(self, fuzzySet: FuzzySet):
        fuzzySet.crisp_set = self.crisp_set
        self.linguistic_terms[fuzzySet.linguistic_term] = fuzzySet

    def plot(self, ax):
        assert len(
            self.crisp_set.dimensions) == 1, "Can only plot fuzzy sets over one variable"
        [name, _] = next(iter(self.crisp_set.dimensions))
        for mf in self.linguistic_terms.values():
            mf.plot(ax)
        ax.set_title(f'Linguistic Variable: {name}')
        ax.set_xlabel(name)
        ax.set_ylabel('Membership Degree')
        ax.legend()

    def __eq__(self, name: str):
        """
        Overload the equality operator to return a linguistic term by name.
        """
        return self.linguistic_terms[name]

    def __repr__(self):
        return f"FuzzyVariable({self.crisp_set}) with sets: [{', '.join(self.linguistic_terms)}]"


def plot3D_surface(input_sets: set[CrispSet], function: callable, axesMap: dict, mesh=15, contour_levels=30):
    fig = plt.figure()
    axs = fig.subplot_mosaic([['A', 'B']])
    fig.set_size_inches(16, 6)

    nameX = axesMap['x']
    nameY = axesMap['y']
    nameZ = axesMap['z']

    setX = make_continuousSet((0, 1))
    setX = make_continuousSet((0, 1))

    # Find the dimensions of the input sets that correspond to the x and y axes
    for iset in input_sets:
        for dim in iset.dimensions:
            if dim[0] == nameX:
                (nameX, setX) = dim
            if dim[0] == nameY:
                (nameY, setY) = dim

    xs = setX.get_coverage(mesh)
    ys = setY.get_coverage(mesh)

    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[function({nameX: x, nameY: y}) for x in xs] for y in ys])

    ss = axs['A'].get_subplotspec()
    axs['A'].remove()
    axs['A'] = fig.add_subplot(ss, projection='3d')
    axs['A'].plot_surface(X, Y, Z, cmap='viridis')
    axs['A'].set_xlabel(nameX)
    axs['A'].set_ylabel(nameY)

    countour = axs['B'].contourf(X, Y, Z, levels=contour_levels)
    axs['B'].set_title
    axs['B'].set_xlabel(nameX)
    axs['B'].set_ylabel(nameY)

    fig.suptitle(f"Surface and Contour plot of {nameZ}")
    fig.colorbar(countour, ax=axs['B'])

    return fig


# In[94]:


class FuzzyRule:
    def __init__(self, antecedent: FuzzySet, consequent: FuzzySet):
        """
        A class representing a fuzzy rule.

        antecedent: The antecedent of the rule. (The condition)
        consequent: The consequent of the rule. (The action)
        """

        self.antecedent = antecedent
        self.consequent = consequent

    def apply(self, data: dict) -> FuzzySet:
        """
        Apply the fuzzy rule to the data.
        Calculates the cut of the antecedent and returns the consequent with the cut applied.

        data: The input data to apply the rule to.
        """
        cut = self.antecedent(data)

        linguistic_term = f"({str(self.consequent)})↑{cut:.2f}"
        crisp_set = self.consequent.crisp_set

        cut_consequent = FuzzySet(linguistic_term, lambda data: min(
            cut, self.consequent(data)), [self.consequent], crisp_set=crisp_set)

        return cut_consequent

    def plot(self, axesMap: dict):
        new_map = axesMap.copy()
        new_map.update({"z": self.antecedent.linguistic_term})
        inputs = set()
        inputs.add(self.antecedent.crisp_set)

        fig = plot3D_surface(inputs, self.antecedent,
                             new_map, mesh=30, contour_levels=30)
        return fig

    def __repr__(self):
        return f"FuzzyRule(\n  IF\t {self.antecedent}\n  THEN\t {self.consequent}\n)"


class FuzzySystem:
    def __init__(self):
        self.rules: list[FuzzyRule] = []
        self.consequent_name = None

    def add_rule(self, rule: FuzzyRule):
        if self.consequent_name is None:
            self.consequent_name = next(
                iter(rule.consequent.crisp_set.dimensions))[0]
        else:
            assert self.consequent_name == next(
                iter(rule.consequent.crisp_set.dimensions))[0], "All consequents must be over the same variable"

        self.rules.append(rule)

    def applyRules(self, data: dict):
        """
        Apply all the rules to the data and return the union of the consequents.
        """

        cut_cut_consequents = [rule.apply(data) for rule in self.rules]

        union = functools.reduce(lambda x, y: x | y, cut_cut_consequents)

        all_based_on = []
        for consequents in cut_cut_consequents:
            all_based_on.extend(consequents.based_on)

        union.based_on = list(set(all_based_on))

        return union

    def predict(self, data: dict):
        """ 
        Apply the rules to the data and return the center of gravity of the union of the consequents.
        This is also the prediction of the fuzzy system for given data

        data: The input data to predict on.
        """

        union = self.applyRules(data)
        (cx, cy) = union.defuzzyfy()
        return cx

    def predictClosest(self, data: dict, algo_ranking: dict[str, float]):
        cx = self.predict(data)

        return cx, min(algo_ranking, key=lambda x: abs(algo_ranking[x] - cx))

    def getInputCrispSets(self):
        inputs = set()
        for rule in self.rules:
            inputs.add(rule.antecedent.crisp_set)
        return inputs

    def __repr__(self):
        newline = "\n"
        return f"FuzzySystem: {self.consequent_name}\n{newline.join(map(str, self.rules))}\n"
