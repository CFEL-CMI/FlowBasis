# FlowBasis: Augmenting basis sets by normalizing flows

The repository contains python implementations of variational solutions to perturbed quantum harmonic oscillator problems. Solution methodology is based on the use of Hermite functions, and augmented Hermite functions, where Hermite functions are composed with normalizing flows [1]. The provided codes reproduces results in the below cited reference.

## Dependencies

Tba.

## Installation

To install the package run:
```
python setup.py install
```
or in a developer-mode, e.g.:
```
python setup.py develop --user
```

## Usage

In HOs.py the Harmonic oscillator problem and the variational methodology to solve it are defined. flows.py contains the normalizing flows that are used to augment standard basis sets. Basis.py contains basis sets that can be used to discretize Schrödinger equations. quadratures.py defines quadrature rules.   


## References

[1] Y. Saleh, A. Iske, A. Yachmenev, J. Küpper, *Augmenting basis sets by normalizing flows*, [arXiv:tba.tba [math.NA]]( tba) (2022).
