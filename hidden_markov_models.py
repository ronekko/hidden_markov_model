# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 19:51:57 2017

@author: sakurai
"""

import numpy as np
from basic_distributions import (
    ProbabilityDistribution,  Multinomial, Gaussian, Poisson
)


class HiddenMarkovModelBase(ProbabilityDistribution):
    def __init__(self, K, p_initial_state, transition_matrix):
        assert K == len(transition_matrix)

        self.K = K
        self.transition_matrix = transition_matrix
        self.p_initial_state = p_initial_state
        self.multinomial_init = Multinomial(p_initial_state)
        multinomials = []
        for k in range(K):
            multinomials.append(Multinomial(transition_matrix[k]))
        self.multinomials = multinomials
        self.components = None

    def __call__(self, num_examples=10000, complete_data=False):
        z_1 = self.multinomial_init(1)[0]
        x_1 = self.components[z_1](1)[0]
        z = [z_1]
        x = [x_1]

        for i in range(1, num_examples):
            z_prev = z[-1]
            z_i = self.multinomials[z_prev](1)[0]
            x_i = self.components[z_i](1)[0]
            z.append(z_i)
            x.append(x_i)

        if complete_data:
            return (np.array(z), np.array(x))
        else:
            return np.array(x)


class GaussianHMM(HiddenMarkovModelBase):
    def __init__(self, K=3, p_initial_state=[0.4, 0.3, 0.3],
                 transition_matrix=[[0.99, 0.009, 0.001],
                                    [0.05, 0.9, 0.05],
                                    [0.001, 0.009, 0.99]],
                 means=[-15, 0, 30], stds=[1, 10, 2]):
        assert K == len(means) == len(stds)
        super(GaussianHMM, self).__init__(
            K, p_initial_state, transition_matrix)

        self.means = means
        self.stds = stds
        self.components = [Gaussian(means[k], stds[k]) for k in range(K)]


class PoissonHMM(HiddenMarkovModelBase):
    def __init__(self, K=3, p_initial_state=[0.4, 0.3, 0.3],
                 transition_matrix=[[0.99, 0.009, 0.001],
                                    [0.05, 0.9, 0.05],
                                    [0.001, 0.009, 0.99]],
                 means=[2, 20, 50]):
        assert K == len(means)
        super(PoissonHMM, self).__init__(
            K, p_initial_state, transition_matrix)

        self.means = means
        self.components = [Poisson(means[k]) for k in range(K)]


if __name__ == '__main__':
    distributions = [GaussianHMM(),  # 0
                     PoissonHMM(),   # 1
                     ]
    dist_type = 1
    sampler = distributions[dist_type]

    x = sampler(10000)

    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())
