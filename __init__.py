# File:                 __init__.py
# Creation date:        2019-11-14
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Implementation of Weighted and Probabilistic Finite State Automata
#
r"""
This package provides a set of tools for Weighted and Probabilistic Finite State Automata. It is based on a purely algebraic view of these models: essentially, an FSA is a list of state transition matrices, one for each symbol. The implementation works with three types of matrices:

* regular 2D arrays of type :class:`numpy.ndarray`
* compressed row sparse matrices of type :class:`scipy.sparse.csr_matrix` and
* a type specifically developed for this purpose, :class:`csra_matrix` which is a subclass of :class:`scipy.sparse.csr_matrix` optimised for deterministic automata.

All the transition matrices of an automaton must be of exactly the same type.

Available types and functions:
------------------------------
:class:`.WFSA`, :class:`.PFSA`, :class:`.Sample`, :class:`.FileBatch`, :class:`.csra_matrix`
"""

from .core import *
from .probabilistic import *
from .util import *
