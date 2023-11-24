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
Script to train normalizing flows for optimizing basis sets to model eigenpairs of petrubed quantum harmonic oscillators."""

import jax.numpy as jnp
import numpy as np
from flowbasis import quadratures, basis, flows
from jax import numpy as jnp, random
import sys
import jax 
import optax 
from jax.config import config 
import h5py as hp 

def setter():
    """setting of global parameters
    """
    global nquads, quads, nmax_local, bases
    if len(nquads) < dim: # we're solving for only a subgroup of the total modes
        for i in range(data["setting"]["dim"]):
            if i not in icoos:
                nquads.insert(i, 1)
                quads.insert(i, quadratures.Herm1d)
                nmax_local.insert(i, None)
                bases.insert(i, None)
            else: pass
    else: pass

def sum_es(params, x):
    """Compues the Hamiltonian matrix
    """
    def _sum_es(params):
        r = QF.apply(params, jnp.array(x), mode="backward")
        grad_QF = flows.jac_x(QF, params, r)#[:,0,0:dim,:][:,:,0:dim]
        det = flows.abs_det_jac_x(QF, params, r).reshape(r.shape[0], -1)
        ddet  = flows.grad_abs_det_jac_x(QF, params, r)
        dpsi_n = jnp.einsum('dij,jdm->mij', dpsi, grad_QF)
        dpsi_n_p = jnp.einsum('lkj,j->lkj', dpsi_n, det[:,0]) 
        dpsi_n_p += jnp.einsum('kj,jm->mkj', psi, ddet[:,0:dim])
        rn = jnp.linalg.norm(r, axis=1)
        poten = (.25*rn**4+0.5*rn**2)
        if compute_matrix: #compute all elements in the matrix 
            v = jnp.einsum('ki,ji,i,i->kj', psi, psi, poten, weights)
            g = jnp.einsum('dki,mji,i,i->kj', dpsi_n_p, dpsi_n_p, weights, 1/(det[:,0]**2))
            h = v + 0.5*g 
        else:
            v = jnp.einsum('ki,ki,i,i->k', psi, psi, poten, weights)
            g = jnp.einsum('dki,mki,i,i->k', dpsi_n, dpsi_n, weights, 1/(det[:,0]**2))
            h = jnp.diag(v + 0.5*g)
        return jnp.trace(h), h 
    return jax.jit(_sum_es)

def solve(): 
    global QF, psi, dpsi, weights
    # 1D quadratures
    x, weights = quadratures.QuadratureGenerator(nquads, quads, wthr=1e-34) 
    shift_QF = (jnp.amax(x, axis=0)+jnp.amin(x, axis=0))/2
    scale_QF = jnp.amax(x-shift_QF, axis=0)/0.99
    QF = flows.mIResNet(NF, scale_QF, shift_QF)
    key1, key2, key3 = random.split(random.PRNGKey(3),3)
    x_dummy = random.uniform(key1, minval=-1, maxval=1, shape=(2,x.shape[1]))
    params = QF.init(key2, x_dummy)
    # primitive basis functions on quadrature grid
    psi, dpsi = basis.BasisGenerator(x, nmax_local, bases, w, nmax_g=nmax_global)
    # Hamiltonian eigenvalues and eigenvectors
    loss = sum_es(params, x)
    loss_grad_fn = jax.value_and_grad(loss, has_aux=True)
    optx = optax.adam(learning_rate=lr)
    opt_state = optx.init(params)
    e = 0
    energies, vectors, losses = [], [], []
    while e<nepochs+1:
        global compute_matrix 
        if e%10==0:
            # compute matrix elements every tenth iteration
            compute_matrix = True
        else:
            compute_matrix = False
        (loss_val, h), grad_val = loss_grad_fn(params)
        updates, opt_state = optx.update(grad_val, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"epoch: {e}, loss: {loss_val}")
        if compute_matrix:
            enr, vec = jnp.linalg.eigh(h)
            print(f"ZPE: {enr[0]}, energy differences: {enr[:20]-enr[0]}")
            energies.append(enr)
            vectors.append(vec)
        # test if the neural network is invertible
        indices = np.random.randint(0, x.shape[0], size=10) 
        y = QF.apply(params, QF.apply(params, jnp.array(x)[indices,:]), mode="backward")
        inversion_error = jnp.abs(y-x[indices,:]).sum()/indices.shape[0]	
        if inversion_error > 1e-4:
            raise ValueError(f"Neural network is not invertible with inversion error: {inversion_error}")
        losses.append(loss_val)
        e += 1
    return energies, vectors, losses


if __name__ == '__main__':
    
    #config.update('jax_disable_jit', True)
    import numpy as np
   
    # define global parameters of the simulation
    global dim, nquads, quads, nmax_local, nmax_global, primitive
    dim = 1 #total dimension of the problem, i.e., number of vibrational modes
    nquads = [60]*dim  # number of quadrature points per dimension 
    quads = [quadratures.Herm1d]*dim # type of quadratures for each mode 
    #nmax_global  = 5 # truncation parameter of multiple D basis
    bases = [basis.Hermite]*dim
    w = [1]*dim # quantum numbers for direct product basis
    nmax_local = [30]*dim 
    # define global parameters of training 
    global nepochs, lr, NF
    nepochs = 100 # number of training epochs 
    lr = 0.001 # learning rate 
    NF = [[24,dim]]
    nmax_global = 3 
    setter()
    energies, vecotrs, losses = solve()
