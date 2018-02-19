# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:08:22 2015

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt


class ProbabilityDistribution(object):
    def get_name(self):
        return self.__class__.__name__

    def get_params(self):
        return self.__dict__

    def visualize(self, data):
        plt.figure()
        self._plot(data)
        plt.figure()
        self._hist(data)
        plt.show()

    def _plot(self, data):
        plt.plot(data, '.')

    def _hist(self, data):
        x_min, x_max = int(min(data)), int(max(data)+1)
        bins = np.arange(x_min, x_max, 0.2)
        plt.hist(data, bins=bins, range=(x_min, x_max))


class Multinomial(ProbabilityDistribution):
    def __init__(self, p=[0.1, 0.2, 0.3, 0.4]):
        self.K = len(p)
        self.p = p

    def __call__(self, num_examples=1000):
        return np.random.choice(self.K, size=num_examples, p=self.p)

    def _hist(self, data):
        bins = np.arange(self.K + 1) - 0.5
        plt.hist(data, bins=bins)
        plt.xticks(range(self.K))


class Gaussian(ProbabilityDistribution):
    def __init__(self, mean=50, std=2):
        self.mean = mean
        self.std = std

    def __call__(self, num_examples=10000):
        return np.random.normal(self.mean, self.std, num_examples)


class Poisson(ProbabilityDistribution):
    def __init__(self, mean=3):
        self.mean = mean

    def __call__(self, num_examples=1000):
        return np.random.poisson(self.mean, num_examples)

    def _hist(self, data):
        x_min, x_max = int(min(data)), int(max(data)+1)
        bins = np.arange(x_min, x_max) - 0.5
        plt.hist(data, bins=bins, range=(x_min, x_max))


class Gamma(ProbabilityDistribution):
    def __init__(self, shape=2, scale=3):
        self.shape = shape
        self.scale = scale

    def __call__(self, num_examples=10000):
        return np.random.gamma(self.shape, self.scale, num_examples)


class Beta(ProbabilityDistribution):
    def __init__(self, alpha=5, beta=1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, num_examples=10000):
        return np.random.beta(self.alpha, self.beta, num_examples)

    def _hist(self, data):
        bins = np.linspace(0, 1, 100)
        plt.hist(data, bins=bins, range=(0, 1))


class LinearModel(ProbabilityDistribution):
    def __init__(self, mean_a=0.001, mean_b=45, std=2):
        self.mean_a = mean_a
        self.mean_b = mean_b
        self.std = std

    def __call__(self, num_examples=10000):
        i = np.arange(num_examples)
        means = self.mean_a * i + self.mean_b
        return np.random.normal(means, self.std, num_examples)


class LinearModel2(ProbabilityDistribution):
    def __init__(self, mean_a=0.01, mean_b=-10, std_a=0.0003, std_b=0.01):
        self.mean_a = mean_a
        self.mean_b = mean_b
        self.std_a = std_a
        self.std_b = std_b

    def __call__(self, num_examples=10000):
        i = np.arange(num_examples)
        means = self.mean_a * i + self.mean_b
        stds = np.exp(self.std_a * i + self.std_b)
        return np.random.normal(means, stds, num_examples)


class PoissonGLM(Poisson):
    def __init__(self, mean_a=0.001, mean_b=1):
        self.mean_a = mean_a
        self.mean_b = mean_b

    def __call__(self, num_examples=1000):
        i = np.arange(num_examples)
        means = np.exp(self.mean_a * i + self.mean_b)
        return np.random.poisson(means)


if __name__ == '__main__':
    distributions = [Multinomial(),   # 0
                     Gaussian(),      # 1
                     Poisson(),       # 2
                     Gamma(),         # 3
                     Beta(),          # 4
                     LinearModel(),   # 5
                     LinearModel2(),  # 6
                     PoissonGLM(),    # 7
                     ]
    dist_type = 0
    sampler = distributions[dist_type]

    x = sampler()

    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())
