import matplotlib.pyplot as plt
import numpy as np
import functools


class CrispSet:
    def __init__(self, dimensions: set[tuple[str, tuple[float, float]]]):
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


def plot3D_surface(input_sets: set[CrispSet], function: callable, axesMap: dict, mesh=15, contour_levels=30):
    fig = plt.figure()
    axs = fig.subplot_mosaic([['A', 'B']])
    fig.set_size_inches(16, 6)

    nameX = axesMap['x']
    nameY = axesMap['y']
    nameZ = axesMap['z']

    rangesX = (0, 1)
    rangesY = (0, 1)

    # Find the dimensions of the input sets that correspond to the x and y axes
    for set in input_sets:
        for dim in set.dimensions:
            if dim[0] == nameX:
                (nameX, rangesX) = dim
            if dim[0] == nameY:
                (nameY, rangesY) = dim

    xs = np.linspace(*rangesX, mesh)
    ys = np.linspace(*rangesY, mesh)

    X, Y = np.meshgrid(xs, ys)
    Z = np.array([[function({nameX: x, nameY: y}) for x in xs] for y in ys])

    ss = axs['A'].get_subplotspec()
    axs['A'].remove()
    axs['A'] = fig.add_subplot(ss, projection='3d')
    axs['A'].plot_surface(X, Y, Z, cmap='viridis')
    axs['A'].set_xlabel(nameX)
    axs['A'].set_ylabel(nameY)
    axs['A'].set_zlabel(nameZ)

    countour = axs['B'].contourf(X, Y, Z, levels=contour_levels)
    axs['B'].set_title
    axs['B'].set_xlabel(nameX)
    axs['B'].set_ylabel(nameY)

    fig.suptitle(f"Surface and Contour plot of {nameZ}")
    colorbar = fig.colorbar(countour, ax=axs['B'])

    annotB = axs['B'].annotate("", xy=(0, 0), xytext=(10, 30),
                               textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="w"),
                               horizontalalignment='center',
                               verticalalignment='center',
                               zorder=1000,
                               arrowprops=dict(arrowstyle="->"))

    colorbar.ax.zorder = -1

    def hover(event):
        if event.inaxes == axs['B']:
            x, y = event.xdata, event.ydata
            z = function({nameX: x, nameY: y})
            annotB.xy = (x, y)
            annotB.set_text(f"(x={x:.2f}, y={y:.2f} z={z:.2f})")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    return fig


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
            assert len(
                self.crisp_set.dimensions) == 1, "Can only evaluate fuzzy sets over one variable"
            (name, _) = next(iter(self.crisp_set.dimensions))
            assert name in data, f"Missing input {name}"
            return self.function(data[name])
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
        (name, range) = next(iter(self.crisp_set.dimensions))

        numX = 0
        numY = 0
        den = 0

        for x in np.linspace(*range, numpoints):
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

        for x in np.linspace(*range, numpoints):
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
        new_linguistic_term = f"({self.linguistic_term} and {
            other.linguistic_term})"
        new_crisp_set = self.crisp_set * other.crisp_set
        return FuzzySet(new_linguistic_term, lambda data: min(self(data), other(data)), [self, other], crisp_set=new_crisp_set)

    def __or__(self, other):
        """
        The union of two fuzzy sets.
        """
        new_linguistic_term = f"({self.linguistic_term} or {
            other.linguistic_term})"
        new_crisp_set = self.crisp_set * other.crisp_set
        return FuzzySet(new_linguistic_term, lambda data: max(self(data), other(data)), [self, other], crisp_set=new_crisp_set)

    def __invert__(self):
        """
        The negation of a fuzzy set.
        """
        new_linguistic_term = f"(not {self.linguistic_term})"
        new_crisp_set = self.crisp_set
        return FuzzySet(new_linguistic_term, lambda data: 1 - self(data), [self], crisp_set=new_crisp_set)

    def plot(self, ax):
        assert len(
            self.crisp_set.dimensions) == 1, "Can only plot fuzzy sets over one variable"
        [name, range] = next(iter(self.crisp_set.dimensions))
        xrange = np.linspace(*range, 1000)

        for mf in self.based_on:
            yh = [mf({name: x}) for x in xrange]
            ax.plot(xrange, yh, label=mf.linguistic_term,
                    linestyle='--', alpha=0.5, linewidth=0.5)

        y = [self({name: x}) for x in xrange]
        ax.plot(xrange, y, label=self.linguistic_term)

        if len(self.based_on) > 0:
            ax.fill_between(xrange, 0, y, alpha=0.25)
            (cog_x, cog_y) = self.defuzzyfy()
            ax.axvline(cog_x, color='black', linestyle='--')
            ax.plot([cog_x], [cog_y], marker='o', markersize=5, color="black",
                    label=f"Center of Gravity: ({cog_x:.2f}, {cog_y:.2f})")

        ax.set_title(f"Linguistic Term: {self.linguistic_term}")
        ax.set_xlabel(name)
        ax.set_ylabel("Membership Value")

        ax.legend()

    def __repr__(self):
        return f"FuzzySet({self.crisp_set} -> {self.linguistic_term})"


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


def makeTriangle(linguistic_term, center, width):
    return Triangle(linguistic_term, center, width)


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


def makeTrapezoid(linguistic_term, left, center_left, center_right, right):
    return Trapezoid(linguistic_term, left, center_left, center_right, right)


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


def makeGaussian(linguistic_term, mean, sigma):
    return Gaussian(linguistic_term, mean, sigma)


class Sigmoid(FuzzySet):
    def __init__(self,  linguistic_term, a, b):
        """
        A class representing a sigmoid fuzzy set.

        linguistic_term: The name of the fuzzy set. (e.g. "young")
        a: The slope of the sigmoid. (The steepness of the sigmoid)
        b: The center of the sigmoid. (The point where the membership value is 0.5)
        """

        self.a = a
        self.b = b
        def function(x): return 1 / (1 + np.exp(-a * (x - b)))
        super().__init__(linguistic_term, function, is_base_set=True)


def makeSigmoid(linguistic_term, a, b):
    return Sigmoid(linguistic_term, a, b)


class LinguisticVariable:
    def __init__(self, name: str, range: tuple[float, float]):
        """
        A class to represent a fuzzy variable.

        crisp_set: The name of the crisp set. (e.g. "age")
        """
        self.crisp_set = CrispSet({(name, range)})
        self.name = name
        self.linguistic_terms: dict[str, FuzzySet] = dict()

    def addLinguisticTerm(self, fuzzySet: FuzzySet):
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

        linguistic_term = f"{self.consequent.linguistic_term}↑{cut:.2f}"
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
        return f"FuzzyRule(IF {self.antecedent} THEN {self.consequent})"


class FuzzySystem:
    def __init__(self):
        self.rules: list[FuzzyRule] = []
        self.consequent: FuzzySet = None

    def addRule(self, rule: FuzzyRule):
        if self.consequent is None:
            self.consequent = rule.consequent
        else:
            assert self.consequent.crisp_set == rule.consequent.crisp_set, "All rules must have the same consequent type"

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
        return f"FuzzySystem with rules:\n\n{newline.join(map(str, self.rules))}"
