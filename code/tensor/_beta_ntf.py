# -*- coding: utf-8 -*-
"""
Copyright © 2012 Telecom ParisTech, TSI
Auteur(s) : Liutkus Antoine
the beta_ntf module is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.
You should have received a copy of the GNU LesserGeneral Public License
along with this program. If not, see <http://www.gnu.org/licenses/>."""

import time
import string
import numpy as np


def _betadiv(a, b, beta):
    if beta == 0:
        return a / b - np.log(a / b) - 1
    if beta == 1:
        return a * (np.log(a) - np.log(b)) + b - a
    return (1. / beta / (beta - 1.) * (a ** beta + (beta - 1.)
            * b ** beta - beta * a * b ** (beta - 1)))


class BetaNTF:
    """BetaNTF class

    Performs nonnegative parafac factorization for nonnegative ndarrays.

    This version implements:
    * Arbitrary dimension for the data to fit (actually up to 25)
    * Any beta divergence
    * Weighting of the cost function

    Parameters
    ----------
    data_shape : the shape of the data to approximate
        tuple composed of integers and of length up to 25.

    n_components : the number of latent components for the NTF model
        positive integer

    beta : the beta-divergence to consider
        Arbitrary float. Particular cases of interest are
         * beta=2 : Euclidean distance
         * beta=1 : Kullback Leibler
         * beta=0 : Itakura-Saito

    n_iter : number of iterations
        Positive integer

    fixed_factors : list of fixed factors
        list (possibly empty) of integers. if dim is in this list, factors_[dim]
        will not be updated during fit.

    Attributes
    ----------
    factors_: list of arrays
        The estimated factors
    """

    # Constructor
    def __init__(self, data_shape, n_components=50, beta=0, n_iter=50,
                 fixed_factors=[], verbose=False, eps=1E-15):
        self.data_shape = data_shape
        self.n_components = n_components
        self.beta = float(beta)
        self.fixed_factors = fixed_factors
        self.n_iter = n_iter
        self.verbose = verbose
        self.eps = eps
        self.factors_= [nnrandn((dim, self.n_components)) for dim in data_shape]

    def fit(self, X, W=np.array([1])):
        """Learns NTF model

        Parameters
        ----------
        X : ndarray with nonnegative entries
            The input array
        W : ndarray
            Optional ndarray that can be broadcasted with X and
            gives weights to apply on the cost function
        """

        eps = self.eps
        beta = self.beta
        ndims = len(self.data_shape)

        print 'Fitting NTF model with %d iterations....' % self.n_iter

        # main loop
        for it in range(self.n_iter):
            if self.verbose:
                if 'tick' not in locals():
                    tick = time.time()
                print ('NTF model, iteration %d / %d, duration=%.1fms, cost=%f'
                       % (it, self.n_iter, (time.time() - tick) * 1000,
                          self.score(X)))
                tick = time.time()

            #updating each factor in turn
            for dim in range(ndims):
                if dim in self.fixed_factors:
                    continue

                # get current model
                model = parafac(self.factors_)

                # building request for this update to use with einsum
                # for exemple for 3-way tensors, and for updating factor 2,
                # will be : 'az,cz,abc->bz'
                request = ''
                operand_factors = []
                for temp_dim in range(ndims):
                    if temp_dim == dim:
                        continue
                    request += string.lowercase[temp_dim] + 'z,'
                    operand_factors.append(self.factors_[temp_dim])
                request += string.lowercase[:ndims] + '->'
                request += string.lowercase[dim] + 'z'
                # building data-dependent factors for the update
                operand_data_numerator = [X * W * (model[...] ** (beta - 2.))]
                operand_data_denominator = [W * (model[...] ** (beta - 1.))]
                # compute numerator and denominator for the update
                numerator = eps + np.einsum(request, *(
                    operand_factors + operand_data_numerator))
                denominator = eps + np.einsum(request, *(
                    operand_factors + operand_data_denominator))
                # multiplicative update
                self.factors_[dim] *= numerator / denominator
        print 'Done.'
        return self

    def score(self, X):
        """Computes the total beta-divergence between the current model and X

        Parameters
        ----------
        X : array
            The input data

        Returns
        -------
        out : float
            The beta-divergence
        """
        return _betadiv(X, parafac(self.factors_), self.beta).sum()

    def __getitem__(self, key):
        """gets NTF model

        First compute the whole NTF model, and then call its
        __getitem__ function. Useful to get the different components. For a
        computationnaly/memory efficient approach, preferably use the
        betaNTF.parafac function

        NTF model is a ndarray of shape 
            self.data_shape+(self.n_components,)
        
        Parameters
        ----------
        key : requested slicing of the NTF model

        Returns
        -------
        ndarray containing the requested slicing of NTF model.
        
        """
        ndims = len(self.factors_)
        request = ''
        for temp_dim in range(ndims):
            request += string.lowercase[temp_dim] + 'z,'
        request = request[:-1] + '->' + string.lowercase[:ndims] + 'z'
        model = np.einsum(request, *(self.factors_))
        return model.__getitem__(key)


def parafac(factors):
    """Computes the parafac model of a list of matrices

    if factors=[A,B,C,D..Z] with A,B,C..Z of shapes a*k, b*k...z*k, returns
    the a*b*..z ndarray P such that
    p(ia,ib,ic,...iz)=\sum_k A(ia,k)B(ib,k)C(ic,k)...Z(iz,k)

    Parameters
    ----------
    factors : list of arrays
        The factors

    Returns
    -------
    out : array
        The parafac model
    """
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.lowercase[temp_dim] + 'z,'
    request = request[:-1] + '->' + string.lowercase[:ndims]
    return np.einsum(request, *factors)


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape

    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))

