#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 100; truncate-lines: t -*-
#
# This file is part of FlowBasis python package
#
# Copyright (C) 2018,2020 CFEL Controlled Molecule Imaging group
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# If you use this program for scientific work, you should correctly reference it; see the LICENSE.md file for details.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see
# <http://www.gnu.org/licenses/>.

"""
An implementation of Hermite basis functions and functions to generate direct product basis sets."""

import numpy as np
from numpy.polynomial.legendre import legval, legder
from numpy.polynomial.hermite import hermval, hermder
import itertools
import math
import opt_einsum
from jax import grad, lax, random, numpy as jnp
import sys 
import jax.numpy as jnp 
import jax 


def Hermite(nmax, r):
    """Normalized Hermite functions and derivatives, i.e., the function outputs:
    f(r) = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) Hn(x)
    df(r)/dr = 1/(pi^(1/4) 2^(n/2) sqrt(n!)) exp(-1/2 x^2) (-Hn(x) x + dHn(x)/dx) * alpha
    where x = (r - r0) alpha
    NOTE: returns f(r) and df(r) without weight factor exp(-1/2 x^2)
  
    Args: 

    - nmax: int, the number of Hermite functions required. 
    - r: ndarray, the points, at which to compute the Hermite functions and derivatives. 
    
    Returns:
     
    f: Hermite functions up to order nmax, evaluated at r. 
    df: Hermite derivatives up to order nmax, evaluated at r.
    """
    x = r
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag([1.0 / np.sqrt(2.0**n * math.factorial(n)) / sqsqpi for n in range(nmax+1)])
    f = hermval(x, c)
    df = (hermval(x, hermder(c, m=1)) - f * x)
    return jnp.array(f), jnp.array(df)

def BasisGenerator(points, nmax_l, bases, w, nmax_g=None, c=None, NF=None):
    """Generates a basis set. If the dimension of the input is bigger than 1, the generated basis set is a direct product of 1-D basis sets. 

    Args:
     
    - points: ndarray, the points at which the basis set is evaluted. 
    - nmax_l: list[int], the order of the basis set in each dimension. 
    - nmax_g: int, order truncation parameter for the direct product basis.
    - basis: list[Callable], the basis sets functions for each dimension.
    - c: Union[list[ndarray], None], Linear parameters that can be applied on 1D basis sets. To be provided if one wants non-primitive basis sets. 
    - NF: normalising flow.
    - w: list[int], weights for computing quantum numbers in the product basis
    
    Returns:

    - basis set evaluated at points.
    """
    psi = []
    dpsi = []
    for i in range(len(nmax_l)):
        if nmax_l[i] != None:
            f, df = bases[i](nmax_l[i], points[:,i])
            if c is not None:
                # need to check this:
                psi.append(jnp.dot(c[icoord][:nmax_l+1, :nmax_l+1], f))
                dpsi.append(jnp.dot(c[icoord][:nmax_l+1, :nmax_l+1], df))
            else:
                psi.append(f)
                dpsi.append(df)          
    if nmax_g == None: nmax_g = nmax_l
    if len(psi) > 1:
        psi_p = prod2(*psi, nmax=nmax_g, w=w)
        dpsi_p = []
        for i in range(len(psi)):
            ket_ = [elem if ielem != i else dpsi[i] for ielem, elem in enumerate(psi)]
            dpsi_p.append(prod2(*ket_, nmax=nmax_g, w=w))
        dpsi_p = jnp.asarray(dpsi_p)
        return psi_p, dpsi_p
    else:
        return psi[0], dpsi[0].reshape(1,psi[0].shape[0],-1)

def prod2(*fn, nmax=None, w=None):
    """Product basis set using einsum

    Args:
        fn : list of arrays (no_funcs, no_points)
            Primitive basis sets, each containing `no_funcs` functions on quadrature grid of `no_points`
            points, note that all basis sets must be defined with respect to the same quadrature grid
        nmax : int
            Maximal value of quantum number in the product basis, defined as
            nmax >= w[0] * n[0] + w[1] * n[1] + w[2] * n[2] ... , where n[0], n[1], n[2] ... are indices
            of functions in each basis set, i.e., n[i] in range(fn[i].shape[0]), and w are weights
        w : list
            Weights for computing quantum number in the product basis

    Returns:
        psi : array (no_funcs, no_points)
            Product basis, containing `no_funcs` product functions on quadrature grid of `no_points` points
    """

    npts = fn[0].shape[1]
    assert (all(f.shape[1] == npts for f in fn)), f"input arrays in `*fn` have different second dimensions: {[f.shape for f in fn]}"
    if nmax is None:
        nmax = max([f.shape[0] for f in fn])
    if w is None:
        w = [1 for i in range(len(fn))]

    psi = fn[0]
    n = jnp.einsum('i,j->ij', [i for i in range(len(fn[0]))], jnp.ones(len(fn[1])))
    nsum = n * w[0]

    for ifn in range(1, len(fn)):
        psi = jnp.einsum('kg,lg->klg', psi, fn[ifn])
        n2 = jnp.einsum('i,j->ij', jnp.ones(len(psi)), [i for i in range(len(fn[ifn]))])
        nsum = nsum + n2 * w[ifn]
        ind = jnp.where(nsum <= nmax)
        psi = psi[ind]
        if ifn <= len(fn)-2:
            nsum = jnp.einsum('i,j->ij', nsum[ind], jnp.ones(fn[ifn+1].shape[0]))
    return psi


