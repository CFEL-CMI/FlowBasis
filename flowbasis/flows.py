#!/usr/bin/env python3
# -*- coding: utf-8; fill-column: 100; truncate-lines: t -*-
#
# This file is part of Active-Learning-of-PES python package
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
An implementation of linear and residual normalizing flows."""

import numpy as np
import functools
import operator
import itertools
import math
import opt_einsum
from jax import grad, lax, random, numpy as jnp
import flax
from flax import linen as nn
from typing import Sequence, Callable, Union
#from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten,
  #                                 Relu, LogSoftmax)
import sys 
import jax.numpy as jnp 
import jax 

def shifted_normal(init_scale, random_scale=1e-2, dtype=jnp.float_):
    """Initializes parameters sampled from a shifted normal distribution.
    
    Args:
    
    - init_scale: float, the center of the normal distribution.
    - random_scale: float, the standard deviation of the normal distribution we want to sample from.
    """
    def init(key, shape, dtype=dtype):
        return random.normal(key, shape, dtype)*random_scale + jnp.asarray(init_scale)#.reshape(-1,1)
    return init

class Unity(nn.Module):
    """A unity neural network, i.e., NN(x) = I(x) = x
    """
    def setup(self):
        pass

    @nn.compact
    def __call__(self, x, mode="whatever"):
        return x

def InvTanh(x):
    return 0.5*jnp.log((1+x)/(1-x))

def LipSwish(x):
    return (x/1.1)*nn.sigmoid(x)

class RegularisedDense(nn.Module):
    """Implements a dense layer that is Lipschitz continuous with Lipschitz constant less than one

    Args:
    
    - features: int, size of the feature that the linear layer maps to
    """
    features: int 
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros 
    
    @nn.compact 
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init, 
                            (inputs.shape[-1], self.features))
        #sv = self._regulariser(kernel)
        s = jnp.linalg.svd(kernel, full_matrices=False, compute_uv=False)
        sv = jnp.max(s)
        def f1(inp): return 0.7*inp[0]/inp[1]
        def f2(inp): return inp[0]
        kernel = jax.lax.cond(sv>=1, true_fun=f1, false_fun=f2, operand=(kernel,sv))
        #kernel = kernel*0.99/sv
        y = jnp.dot(inputs, kernel)
        bias = self.param('bias', self.bias_init, (self.features,))
        y = y + bias 
        return y 
     
class InvBlock(nn.Module):
    """An invertible block composed of a linear layer and a nonlinear LipSwish activation function.

    Args:
     
    - features: list[int], the number of features in every layer
    """
    features: Sequence[int]

    def setup(self):
        self.DenseLayers = [RegularisedDense(feat) for feat in self.features]

    @nn.compact
    def __call__(self,x):
        for i, lyr in enumerate(self.DenseLayers):
            x = lyr(x)
            x = LipSwish(x)
        return x

class LinearFlow(nn.Module):
    """Implements NN(r) = (r-ref)*alpha where alpha is a learnable parameter 
    
    Args:
   
    - init_ref: Sequence[float], the mean value of the Gaussian, from which we sample the initial value of ref.
    - init_scale: Sequence[float], the mean value of the Gaussian, from which we sample the initial value of alpha.
    - random_scale: float, the standard deviation of the Gaussian, from which we sample the values of alpha and ref
    """
    init_ref: Sequence[float]
    init_scale: Sequence[float]
    random_scale: float
    
    def setup(self):
        self.kernel_init = shifted_normal(self.init_scale, self.random_scale)
        self.ref_init = shifted_normal(self.init_ref, self.random_scale)
  
    @nn.compact 
    def __call__(self, X, mode="forward"):
        kernel = self.param('kernel', 
                           self.kernel_init, 
                           (1,X.shape[-1])).T
        shift = self.param('shift', 
                           self.ref_init, 
                           (1,X.shape[-1]))
        if mode == "forward":
            return  jnp.einsum('ij,jk->ij', (X-shift), kernel)   
        else:
            try:
                return jnp.einsum('ij,jk->ij', X, 1/kernel) + shift
            except:
                return X*(1/kernel[0]) + shift[0]

class Inverse(nn.Module):
    @nn.compact
    def __call__(self, x, x0, dense):
        x = x0 - dense(x)
        return x, x

class IResNet(nn.Module):
    """An invertible Resnet: NN(x) = I + g(x) where g is an InvBlock model 
    
    Args:
     
    - features: list[int], the number of features in every layer
    """
    features: Sequence[int]

    def setup(self):
       self.IBlock = InvBlock(self.features)
    
    @nn.compact
    def __call__(self, X, mode="forward"):
       input = X
       if mode == "forward":
           x = self.IBlock(input)
           x = x + input
       elif mode == "backward":
           x0 = input
           units = nn.scan(Inverse, variable_broadcast="params", split_rngs={"params": True}, in_axes=0)
           x, _ = units()(input, jnp.array([x0]*1000), self.IBlock)
       return x

class mIResNet(nn.Module):
    """Generates a sequence of Invertible resnets defined on [-1,1]. InvTanh and Tanh functions are applied to the input and output to guarantee that the domain and range of the model is [-1,1]. Additionally, the model allows for mapping over other ranges by controllingfixed, i.e., non-learnable, scale and shift parameters.

    Args:

    - flows_init: Sequence[Sequence], defines the number of layers in every normalising flow.
    - scale: float or None, if float, it indicates the scaling required to map the range of the input [a,b] to [-1,1]. If None, the input is supposed to be defined on [-1,1].
    - shift: float or None, if float, it indicates the shift required to map the range of the input [a,b] to [-1,1]. If None, the input is supposed to be defined on [-1,1].
    """
    flows_init: Sequence[Sequence]
    scale: Union[float, None]
    shift: Union[float, None]
    
    def setup(self):
        self.n_flows = len(self.flows_init)
        self.flows = [IResNet(self.flows_init[i]) for i in range(self.n_flows)]
    
    @nn.compact 
    def __call__(self, X, mode="forward"):
        if self.scale is not None:
            # we are mapping [a,b] to [-1,1]
            X = (X-self.shift)/self.scale
        if mode=="forward":
            X = InvTanh(X)
            for i in range(self.n_flows):
                X = self.flows[i](X)
            X = nn.tanh(X)
        else:
            X = InvTanh(X)
            for i in range(self.n_flows):
                X = self.flows[-1-i](X, mode="backward")
            X = nn.tanh(X)
        if self.scale is not None:
            return X*self.scale + self.shift
        else:
            return X

class LinearInvBlock(nn.Module):
    """The following model concatenates LinFlow with an mIResNet. 
    
    Args:

    See Args of mIResNet and LinearFlow.
    icoos: sequence of which cooridnates to apply mIResNet on
    """
    ref: Sequence[float]
    init_scale: Sequence[float]
    random_scale: float
    features: Sequence[int] 
    scale: Sequence[float]
    shift: Sequence[float]
    icoos: Sequence[int]
 
    def setup(self):
        self.IResNet = mIResNet(self.features, self.scale[jnp.array(self.icoos)], self.shift[jnp.array(self.icoos)])
        self.LinearFlow = LinearFlow(self.ref, self.init_scale, self.random_scale)
    
    @nn.compact 
    def __call__(self, X, mode="forward", nonlinear=True):
        if mode == "forward":
           y = self.LinearFlow(X)
           y_nl = self.IResNet(y[:,self.icoos])
           y = y.at[:,self.icoos].set(y_nl)
           return y
        elif mode == "backward":
            y_nl = self.IResNet(X[:,self.icoos], mode="backward")
            X = X.at[:,self.icoos].set(y_nl)
            y = self.LinearFlow(X, mode="backward")
            return y

# the followings are functions to compute various kinds of derivatives of NN-models

def jac_x(model, params, x_batch, **kwargs):
    def jac(params):
        def jac(x):
            return jax.jacrev(model.apply, 1)(params, x, **kwargs)
        return jax.vmap(jac, in_axes=0)(x_batch)
    return jax.jit(jac)(params)

def hess_x(model, params, x_batch, **kwargs):
    def hess(params):
        def hess(x):
            return jax.jacfwd(jax.jacrev(model.apply, 1), 1)(params, x, **kwargs)
        return jax.vmap(hess, in_axes=0)(x_batch)
    return jax.jit(hess)(params)

def abs_det_jac_x(model, params, x_batch, **kwargs):
    def det(params):
        def det(x):
            return jnp.sqrt(jnp.abs(jnp.linalg.det(jax.jacrev(model.apply, 1)(params, x, **kwargs))))
        return jax.vmap(det, in_axes=0)(x_batch)
    return jax.jit(det)(params)

def grad_abs_det_jac_x(model, params, x_batch, **kwargs):
    def grad(params):
        def det(x):
            try:    
                return jnp.sqrt(jnp.abs(jnp.linalg.det(jax.jacrev(model.apply, 1)(params, x, **kwargs))))[0]
            except: # this works with unity flows
                return jnp.sqrt(jnp.abs(jnp.linalg.det(jax.jacrev(model.apply, 1)(params, x, **kwargs))))
 
                return
        return jax.vmap(jax.grad(det), in_axes=0)(x_batch)
    return jax.jit(grad)(params)


# NOTE: implement hessian for diagonal elements only
def hess_abs_det_jac_x(model, params, x_batch, **kwargs):
    def hess(params):
        def det(x):
            return jnp.abs(jnp.linalg.det(jax.jacrev(model.apply, 1)(params, x, **kwargs)))
        return jax.vmap(jax.hessian(det), in_axes=0)(x_batch)
    return jax.jit(hess)(params)

if __name__ == '__main__':
    pass
