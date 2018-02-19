# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 22:05:07 2018

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans

from hidden_markov_models import GaussianHMM


def col(x):
    '''Columnize a 1-dim ndarray. When x.shape == (N,), this function returns
    (N, 1) shaped x.
    '''
    return x[:, None]


# Estimate parameters of HMM with Gaussian emission.
# Emission distributions are Gaussian with known variance 1
if __name__ == '__main__':
    data_file = 'hmm_data.npy'
    use_data_file = False
    show_data = True

    K = 3
    N = 10000
    iter_initial_gmm = 1
    iter_hmm = 5000
    term_threshold = 1e-12
    use_kmeans_init = False

#    true_dist = GaussianHMM()
    true_dist = GaussianHMM(transition_matrix=[[0.99, 0.005, 0.005],
                                               [0.01, 0.98, 0.01],
                                               [0.005, 0.005, 0.99]],
                            means=[-2, 0, 2],
                            stds=[0.5, 10, 0.5])
    if use_data_file:
        x = np.load(data_file)
    else:
        z, x = true_dist(N, True)
        np.save(data_file, x)
    if show_data:
        true_dist.visualize(x)

    # Estimate initial parameters of emission distributions as GMM by EM
    x = col(x)

    # initialize gamma = p(z_i|x_i)
    if use_kmeans_init:
        kmeans = KMeans(K)
        z = kmeans.fit_predict(x)
        gamma = np.zeros((N, K))
        gamma[range(N), z] = 1
    else:
        gamma = np.random.dirichlet(np.ones(K), N)

    log_likelihoods = []
    for it in range(iter_initial_gmm):
        N_k = gamma.sum(0)
        means = (gamma * x).sum(0) / N_k
        var = (gamma * ((x - means) ** 2)).sum(0) / N_k
        std = np.sqrt(var)
        weights = N_k / N
        likelihood = weights * stats.norm(means, std).pdf(x)  # f(x|z, params)
        mass = likelihood.sum(1, keepdims=True)
        gamma = likelihood / mass

        # show plots
        dummy = np.linspace(-10, 10, 600)
        f = weights * stats.norm(means, std).pdf(col(dummy))
        plt.plot(dummy, f)
        plt.show()
        log_likelihoods.append(np.sum(np.log(mass)))
        print(it)
        print('weights:\n', weights)
        print('means:\n', means)
        print('std:\n', std)
        print('likelihood:', log_likelihoods[-1])

        if it > 1:
            improvement = log_likelihoods[-1] - log_likelihoods[-2]
            print(improvement)
            if 0 <= improvement < term_threshold:
                break


    # renumber the state ids according to means in ascending order
    perm = means.argsort()
    gamma = gamma[:, perm]
    means = means[perm]
    std = std[perm]
    weights = weights[perm]
    print('initial means:', means)

    # Estimate HMM parameters using initial parameters estimated before
    pi = true_dist.p_initial_state
#    A = np.random.dirichlet(np.ones(K), K)  # randomly initialied A
    xi = gamma[:-1, :, None] * gamma[1:, None]
    A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
    print('A:\n', A)

    log_likelihoods = []
    for it in range(iter_hmm):
        # Estimate gamma by forward-backward algo.
        alphas = []
        betas = []
        cs = []

        # precompute p(x_n|z_n)
        pxngzn = stats.norm(means, std).pdf(x)

        # forwrd (alpha)
        alpha = pxngzn[0] * pi
        c = alpha.sum()
        alpha /= c
        alphas.append(alpha)
        cs.append(c)
        for pn in pxngzn[1:]:
            alpha = pn * alpha.dot(A)
            c = alpha.sum()
            alpha /= c
            alphas.append(alpha)
            cs.append(c)
        alphas = np.array(alphas)

        # backward (beta)
        beta = np.ones(K)
        betas.append(beta)
        for pn, c in zip(pxngzn[1:][::-1], cs[1:][::-1]):
            beta = (beta * (pn / c)).dot(A.T)
            betas.append(beta)
        betas = np.array(betas[::-1])

        # gamma
        gamma = alphas * betas
#        print(z[-10:])
#        print(gamma2[-10:])
        log_likelihood = np.sum(np.log(cs))
        log_likelihoods.append(log_likelihood)

        # Estimate parameters with gamma
        N_k = gamma.sum(0)
        means = (gamma * x).sum(0) / N_k
        var = (gamma * ((x - means) ** 2)).sum(0) / N_k
        std = np.sqrt(var)

        # xi
        alphas_expand = alphas[:-1, :, None]  # (N, K) -> (N - 1, K, 1)
        tmp = (
            pxngzn / np.expand_dims(cs, 1) * betas)
        tmp_expand = tmp[1:, None]  # (N, K)->(N - 1, 1, K)
        xi = alphas_expand * tmp_expand * A
        A = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)

        # show plots
        dummy = np.linspace(-15, 15, 600)
        f = stats.norm(means, std).pdf(col(dummy))
        if it % 10 == 0:
            plt.plot(dummy, f)
            plt.show()
            plt.plot(log_likelihoods)
            plt.show()
            print(it)
            print('log likelihood:', log_likelihood)
            print('means:\n', means)
            print('std:\n', std)
            print('A:\n', A)
            print()

            if it > 1:
                improvement = log_likelihoods[-1] - log_likelihoods[-2]
                print(improvement)
                if 0 <= improvement < term_threshold:
                    break
