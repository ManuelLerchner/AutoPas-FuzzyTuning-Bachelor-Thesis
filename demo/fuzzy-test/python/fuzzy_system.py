#!/usr/bin/env python
# coding: utf-8

# In[84]:


from abc import abstractmethod
import matplotlib.pyplot as plt
import matplotlib as mpl
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

    def get_coverage(self):
        if self.type.name == Set.SetType.CONTINUOUS.name:
            return (self.values[0], self.values[1])
        elif self.type.name == Set.SetType.DISCRETE.name:
            raise ValueError("Invalid Set Type")
            return (self.values[0], self.values[-1])
        else:
            raise ValueError("Invalid Set Type")

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

                if name not in data:
                    raise ValueError(f"Variable {name} not found in data")

                return self.function(data[name])
            except:
                return self.function(data)
        else:
            # Function is a function a lambda function combining other fuzzy sets
            # Pass the data down recursively
            return self.function(data)

    @abstractmethod
    def getIntegrationRange(self):
        coverage = []
        for fs in self.based_on:
            c = fs.getIntegrationRange()
            coverage.extend(c)

        # merge intervals
        coverage.sort(key=lambda x: x[0])
        merged = []
        for c in coverage:
            if not merged or merged[-1][1] < c[0]:
                merged.append(list(c))
            else:
                merged[-1][1] = max(merged[-1][1], c[1])

        return merged

    def defuzzyfy(self, method, n):
        if method == "mom":
            return self.mom(n)
        elif method == "cog":
            return self.cog(n)
        else:
            raise ValueError("Invalid defuzzification method")

    def cog(self, n):
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

        (s, e) = cset.get_coverage()
        x_range = np.linspace(s, e, n)

        self.check_enough_coverage(s, e, n)

        for x in x_range:
            y = self.function({name: x})
            numX += x*y
            numY += 0.5*y*y
            den += y

        if den == 0:
            raise ValueError("No area under the curve")
            return (0, 0)

        return (numX/den, numY/den)

    def mom(self, n):
        """
        Calculate the mean of maximum of the fuzzy set.

        Returns: (x,y) The x and y coordinates of the mean of maximum.
        Uses the range of the crisp set to calculate the mean of maximum numerically.
        """

        assert len(
            self.crisp_set.dimensions) == 1, "Can only calculate mean of maximum for one variable"
        (name, range) = next(iter(self.crisp_set.dimensions))

        results = []
        max_y = 0

        places_of_max = []

        (s, e) = range.get_coverage()
        x_range = np.linspace(s, e, n)

        self.check_enough_coverage(s, e, n)

        for x in x_range:
            y = self.function({name: x})
            if y > max_y:
                max_y = y
                places_of_max = [x]
            elif y == max_y:
                places_of_max.append(x)

        if len(places_of_max) == 0:
            raise ValueError("No maximum found")

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

    def check_enough_coverage(self, s, e, n):
        membership_coverage = self.getIntegrationRange()
        delta_x = (e-s)/n

        if delta_x == 0:
            return

        for (a, b) in membership_coverage:

            # warning if delta_x is too large. At least 50 points per interval
            MIN_POINTS = 10
            if (b-a)/delta_x < MIN_POINTS:
                print(f"Warning: Not enough points in the interval ({a}, {b})")
                print(f"delta_x = {delta_x}, number of points = {
                      (b-a)/delta_x}, should be at least {MIN_POINTS}")
                # raise ValueError("Not enough points in the interval")

    def plot(self, ax, n=100, defuzzifiationMethod="mom"):
        assert len(
            self.crisp_set.dimensions) == 1, "Can only plot fuzzy sets over one variable"
        [name, set] = next(iter(self.crisp_set.dimensions))

        (s, e) = set.get_coverage()
        xrange = np.linspace(s, e, n)

        self.check_enough_coverage(s, e, n)

        for mf in self.based_on:
            [_, mf_set] = next(iter(mf.crisp_set.dimensions))
            yh = [mf({name: x}) for x in xrange]

            if mf_set.type == Set.SetType.DISCRETE:
                total_range = max(mf_set.values) - min(mf_set.values)
                ax.bar(xrange, yh, label=None,
                       width=total_range/(50*len(mf_set.values)), alpha=0.5)
            else:
                ax.plot(xrange, yh, label=None,
                        linestyle='--', alpha=0.5, linewidth=0.5)

        y = [self({name: x}) for x in xrange]

        if set.type == Set.SetType.DISCRETE:
            total_range = max(set.values) - min(set.values)
            ax.bar(xrange, y, label=str(self),
                   width=total_range/(50*len(set.values)))
        else:
            # self.crisp_set = None
            ax.plot(xrange, y, label=str(self))

        if len(self.based_on) > 0:
            if set.type == Set.SetType.CONTINUOUS:
                ax.fill_between(xrange, 0, y, alpha=0.25)
            (cog_x, cog_y) = self.defuzzyfy(method=defuzzifiationMethod, n=n)
            ax.axvline(cog_x, color='black', linestyle='--')
            ax.plot([cog_x], [cog_y], marker='o', markersize=5, color="black",
                    label=f"Defuzzified Value: ({cog_x:.2f}, {cog_y:.2f}) [method={defuzzifiationMethod}]")

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

    def getIntegrationRange(self):
        return [(self.center - self.width, self.center + self.width)]

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Triangle({self.center}, {self.width})"


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

    def getIntegrationRange(self):
        return [(self.left, self.right)]

    def __repr__(self):
        prefix = ""
        # if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
        #     prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Trapezoid({np.round(self.left, 0)}, {np.round(self.center_left, 0)}, {np.round(self.center_right, 0)}, {np.round(self.right, 0)})"


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

    def getIntegrationRange(self):
        return [(self.mean - 3*self.sigma, self.mean + 3*self.sigma)]

    def __repr__(self):
        prefix = ""
        # if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
        #     prefix = next(iter(self.crisp_set.dimensions))[0] + " is "
        return f"{prefix}\"{self.linguistic_term}\": Gaussian({np.round(self.mean, 0)}, {np.round(self.sigma, 0)})"


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

    def getIntegrationRange(self):
        return [(self.center - 3*abs(self.width), self.center + 3*abs(self.width))]

    def __repr__(self):
        prefix = ""
        # if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
        #     prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Sigmoid({np.round(self.center, 0)}, {np.round(self.width, 2)})"


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

    def getIntegrationRange(self):
        return [(min(self.dm, self.dn), max(self.dm, self.dn))]

    def peak(self):
        return np.inf if self.dn > self.dm else -np.inf

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": SigmoidFinite({self.dm}, {self.beta}, {self.dn})"


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

    def getIntegrationRange(self, n):
        return [(self.value, self.value)]

    def __repr__(self):
        prefix = ""
        if (self.crisp_set and len(self.crisp_set.dimensions) == 1):
            prefix = str(next(iter(self.crisp_set.dimensions))[0]) + " is "
        return f"{prefix}\"{self.linguistic_term}\": Singleton({self.value})"


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

    def plot(self, ax, n=100):
        assert len(
            self.crisp_set.dimensions) == 1, "Can only plot fuzzy sets over one variable"
        [name, _] = next(iter(self.crisp_set.dimensions))
        for mf in self.linguistic_terms.values():
            mf.plot(ax, n=n)
        ax.set_title(f'Linguistic Variable: {name}')
        ax.set_xlabel(name)
        ax.set_ylabel('Degree of Membership')
        ax.legend(loc='lower right')

        # ax.axhline(0, color='black', linewidth=1)

    def __eq__(self, name: str):
        """
        Overload the equality operator to return a linguistic term by name.
        """
        return self.linguistic_terms[name]

    def __repr__(self):
        return f"FuzzyVariable({self.crisp_set}) with sets: [{', '.join(self.linguistic_terms)}]"


def plot3D_surface(input_sets: set[CrispSet], function: callable, axesMap: dict, labelMap: dict = {}, n=100, contour_levels=30, fixed_values={}):
    fig = plt.figure()
    axs = fig.subplot_mosaic([['A', 'B']])
    fig.set_size_inches(16, 6)

    nameX = axesMap['x']
    nameY = axesMap['y']
    nameZ = axesMap['z']

    setX = make_continuousSet((0, 1))
    setX = make_continuousSet((0, 1))
    setY = make_continuousSet((0, 1))

    # Find the dimensions of the input sets that correspond to the x and y axes
    for iset in input_sets:
        for dim in iset.dimensions:
            if dim[0] == nameX:
                (nameX, setX) = dim
            if dim[0] == nameY:
                (nameY, setY) = dim

    xrange = setX.get_coverage()
    xs = np.linspace(xrange[0], xrange[1], n)

    yrange = setY.get_coverage()
    ys = np.linspace(yrange[0], yrange[1], n)

    X, Y = np.meshgrid(xs, ys)
    Z = np.array(
        [[function({nameX: x, nameY: y, **fixed_values}) for x in xs] for y in ys])

    ss = axs['A'].get_subplotspec()
    axs['A'].remove()
    axs['A'] = fig.add_subplot(ss, projection='3d')
    axs['A'].plot_surface(X, Y, Z, cmap='viridis')
    axs['A'].set_xlabel(nameX)
    axs['A'].set_ylabel(nameY)

    axs['B'].set_title
    axs['B'].set_xlabel(nameX)
    axs['B'].set_ylabel(nameY)

    fig.suptitle(f"Surface and Contour plot of {nameZ}")
    fig.text(0.3, 0.04, fixed_values, ha='center')

    if labelMap:
        norm_bins = sorted(labelMap.keys())
        norm_values = [labelMap[x] for x in norm_bins]

        norm = mpl.colors.BoundaryNorm(
            norm_bins, len(norm_bins)+1, extend='max')

        countour = axs['B'].pcolormesh(X, Y, Z, norm=norm, cmap='tab20')

        cbar = fig.colorbar(countour, ax=axs['B'], norm=norm)

        cbar.set_ticks(norm_bins)
        cbar.ax.set_yticklabels(norm_values)
    else:
        countour = axs['B'].contourf(X, Y, Z, levels=contour_levels)
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
                             new_map, n=100, contour_levels=30)
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

    def predict(self, data: dict, n=100, method="mom"):
        """ 
        Apply the rules to the data and return the center of gravity of the union of the consequents.
        This is also the prediction of the fuzzy system for given data

        data: The input data to predict on.
        """

        union = self.applyRules(data)
        (meanX, y) = union.defuzzyfy(method, n=n)
        return meanX

    def predictClosest(self, data: dict, algo_ranking: dict[str, float], n=100, method="mom"):
        cx = self.predict(data, n=n, method=method)

        closest = min(algo_ranking, key=lambda x: abs(algo_ranking[x] - cx))

        return cx, closest

    def getInputCrispSets(self):
        inputs = set()
        for rule in self.rules:
            inputs.add(rule.antecedent.crisp_set)
        return inputs

    def __repr__(self):
        newline = "\n"
        return f"FuzzySystem: {self.consequent_name}\n{newline.join(map(str, self.rules))}\n"
