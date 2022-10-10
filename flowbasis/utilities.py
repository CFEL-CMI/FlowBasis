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

"""Plotting and IO utilities."""

import h5py as hp
import matplotlib
import json 
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
from scipy.stats import linregress

def convergence_check(filenames):
    """plot all the energies as a function of N to see whether or not they converge
    """
    colors = ['b', 'r', 'orange', 'darkcyan', 'green', 'indigo']
    for filename in filenames:
        with hp.File('simulations/'+filename, 'r') as of:
            for j in range(0,30,5):
                energies, Ns = [], []
                for i in range(j+5, 30):
                    energy = np.asarray(of.get('nmax_local='+str(i)+"/energies"))[-1,:]
                    Ns.append(len(energy)) 
                    energies.append(energy[j:j+5].sum())
                Ns, energies = np.asarray(Ns), np.asarray(energies)
                if 'NF' in filename: linestyle='solid' 
                else: linestyle='dashed'
                plt.plot(Ns, energies, marker="*", markersize=8, linestyle=linestyle, c=colors[int(j/5)])
    plt.yscale('log')
    plt.xlabel('$\it{N}$')
    plt.ylabel('Bands of 5 eigenvalues (log scale)')
    #plt.show()
    plt.savefig('simulations/1D_comparison.pdf', dpi=500, bbox_inches='tight')

def rate_convergence(filenames):
    """plot all the energies as a function of N to see whether or not they converge
    """
    for filename in filenames:
        with hp.File('simulations/'+filename, 'r') as of:
            energies, Ns = [], []
            for i in range(10, 30):
                energy = np.asarray(of.get('nmax_local='+str(i)+"/energies"))[-1,:]
                Ns.append(len(energy)) 
                energies.append(energy[5:10].sum())
                print(filename)
                if i == 29:
                    energy_converged = energy[5:10].sum()
            Ns, energies = np.asarray(Ns), np.asarray(energies)
        convergence = []    
        for i in range(0, len(Ns)-1):
            convergence.append(np.abs(energies[i+1]-energy_converged)/np.abs(energies[i]-energy_converged))
  
        if 'NF' in filename: linestyle='solid' 
        else: linestyle='dashed'
        # linear regression 
        slope, intercept, r, p, se = linregress(Ns[1:], convergence)
        #plt.plot(Ns[1:], convergence, marker="*", markersize=8, linestyle=linestyle)
        if 'standard' in filename: label='Hermite functions'
        else: label = 'Augmented Hermite functions'
        plt.plot(Ns[1:], intercept+slope*Ns[1:], marker="*", markersize=8, color='darkcyan', linestyle=linestyle, label=label)
       
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('$\it{N}$ (log scale)')
    plt.ylabel('convergence rate (log scale)')
    plt.show()
    #plt.savefig('simulations/1D_comparison_convergence_rate.pdf', dpi=500, bbox_inches='tight')


def plotter(filenames):
    # read real data:
    with hp.File('simulations/1D_standard_calcs.h5', 'r') as of:
        real_energies = np.asarray(of.get('nmax_local=29/energies'))[0]
    for filename in filenames:
        with hp.File('simulations/'+filename, 'r') as of:
            error, Ns = [], []
            for i in range(20, 29):
                energy = np.asarray(of.get('nmax_local='+str(i)+"/energies"))[-1,:]
                Ns.append(len(energy))
                error.append(np.abs(energy[0:16]-real_energies[0:16])/real_energies[0:16])
        error, Ns = np.asarray(error).sum(axis=1)/16, np.asarray(Ns)
        plt.plot(Ns, error, marker=".",label=filename)
        #plt.yscale('log')
    plt.legend()
    plt.show()

         
if __name__ == '__main__':
    #plotter(['2D_standard_calcs.h5', '2D_NF_calcs.h5'])
    #convergence_check(['1D_standard_calcs.h5', '1D_NF_calcs.h5'])
    rate_convergence(['1D_NF_calcs.h5','1D_standard_calcs.h5']) 

