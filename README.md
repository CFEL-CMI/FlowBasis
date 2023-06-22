# FlowBasis: Augmenting basis sets by normalizing flows

The repository contains python implementations of variational solutions to perturbed quantum harmonic oscillator problems. Solution methodology is based on the use of Hermite functions, and augmented Hermite functions, where Hermite functions are composed with normalizing flows [1]. The provided codes reproduces results in the below cited reference.

## Dependencies

The codes were tested on Python 3.6.8. We recommend the use of a virtual
environment. To install all necessary packages run:

```
pip install -r requirements.txt
```


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

In examples/HOs.py the Harmonic oscillator problem and the variational methodology to solve it are defined. flowbasis/flows.py contains the normalizing flows that are used to augment standard basis sets. flowbasis/Basis.py contains basis sets that can be used to discretize Schrödinger equations. flowbasis/quadratures.py defines quadrature rules.   


## References

[1] Y. Saleh, A. Iske, A. Yachmenev, J. Küpper, *Augmenting basis sets by normalizing flows*, [Proc. Appl. Math. Mech. 23 (1) (2023), e202200239]( https://onlinelibrary.wiley.com/doi/full/10.1002/pamm.202200239) (2022).
