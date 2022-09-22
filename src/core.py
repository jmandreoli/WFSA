# File:                 core.py
# Creation date:        2019-11-14
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Implementation of Weighted Finite State Automata
#

import logging; logger = logging.getLogger(__name__)
from typing import Optional, Union, Callable, Dict, Any, Sequence
from functools import cached_property
from numpy import ndarray, all, sum, zeros, ones, unique, amin, amax, empty, einsum
from scipy.sparse import csr_matrix, vstack as spvstack
from .util import WFSABrowsePlugin, lmatrix, csra_matrix, onehot

__all__ = 'WFSA',

#===============================================================================
class WFSA (WFSABrowsePlugin):
#===============================================================================
  r"""
An instances of this class is a Weighted Finite State Automaton, specified as an assignment of a transition matrix to each symbol in an alphabet. No initial nor final state is specified at that point.

WFSA intersection is supported through operator ``@``. It is essentially commutative, except for the state names, hence the choice of ``@`` instead of ``*``: state names of the intersection are composed by joining names from the operands with ``.``.

An additional transition matrix (called a template) can optionally be specified in a WFSA. It is used when intersecting automata with different alphabets. Each time a symbol is present in one operand, but not in the other, the template of the second operand, if present, is used as default transition matrix for that symbol in the second operand. Also, if both operands have a template, the intersection also has a template, computed as for the other transition matrices.

Methods:
  """

  mtype: type
  r"""The common type of all the transition matrices (one of the supported types)"""
  N: int
  r"""The number of states"""
  n: int
  r"""The number of symbols in the alphabet"""
  W: Union[tuple[ndarray,...],tuple[csr_matrix,...],tuple[csra_matrix,...]]
  r"""a list of transition (square) matrices with the same shape and type, one for each symbol in the alphabet"""
  template: Optional[Union[ndarray,csr_matrix,csra_matrix]]
  r"""a transition (square) matrix of same shape and type as those in :attr:`W`"""
  symb_names: tuple[str,...]
  r"""the list of names of the symbols (indices in the list of transition matrices)"""
  state_names: tuple[str,...]
  r"""the list of names of the states (indices in each transition matrix)"""

#-------------------------------------------------------------------------------
  def __init__(self,*a,**ka):
#-------------------------------------------------------------------------------
    self.init_struct(*a,**ka)
    self.init_type()

#-------------------------------------------------------------------------------
  def init_struct(self,W:Union[Sequence[ndarray],Sequence[csr_matrix],Sequence[csra_matrix]],template:Optional[Union[ndarray,csr_matrix,csra_matrix]]=None,state_names:Optional[Sequence[str]]=None,symb_names:Optional[Sequence[str]]=None,check:bool=True):
    r"""Called at initialisation with all the arguments of the constructor. Initialises the structure of this WFSA. If *check* is :const:`True`, a number of consistency checks are performed on the arguments (set to :const:`False` if already checked at invocation)."""
#-------------------------------------------------------------------------------
    W = tuple(W)
    base = W[0] if template is None else template
    if check:
      N,N_ = base.shape
      if not isinstance(base,(ndarray,csr_matrix)): raise ValueError(f'Unsupported matrix type {base.__class__}')
      if any([type(w)!=type(base) for w in W]): raise ValueError('Transition matrices must all be of the same type')
      if N!=N_ or any([w.shape!=(N,N) for w in W]): raise ValueError('Transition matrices must all be of the same square shape')
      for c,w in enumerate(W):
        if amin(w)<0.: raise ValueError(f'Negative transition weight in symbol: {c}')
      if template is not None and amin(template)<0.: raise ValueError('Negative transition weight in template')
      if state_names is not None and (len(state_names)!=N or not all([isinstance(x,str) for x in state_names])): raise ValueError('State name list must contain strings and match state dimension')
      if symb_names is not None and (len(symb_names)!=len(W) or not all([isinstance(x,str) for x in symb_names])): raise ValueError('Symbol name list must contain strings and match alphabet length')
    self.W = W
    self.template = template
    self.n = len(W)
    self.N = base.shape[0]
    self.mtype = type(base)
    self.state_names = self.default_state_names() if state_names is None else tuple(state_names)
    self.symb_names = self.default_symb_names() if symb_names is None else tuple(symb_names)

#-------------------------------------------------------------------------------
  def init_type(self):
    r"""Initialises the types of this WFSA"""
#-------------------------------------------------------------------------------
    # Initialisation of special methods dependent on matrix type
    if issubclass(self.mtype,ndarray):
      def initial(a,size,N=self.N): r = empty((size,N)); r[...] = a[None,:]; return r
      def max_normalise(m):
        x = amax(m,axis=1,keepdims=True)
        x[x==0] = 1; m /= x
        return x
      product = lambda w1,w2: einsum('ij,kl->ikjl',w1,w2).reshape(2*(w1.shape[0]*w2.shape[0],))
    elif issubclass(self.mtype,csr_matrix):
      if issubclass(self.mtype,csra_matrix):
        factory = csra_matrix.initial
        def initial(a,size,N=self.N,factory=factory): return factory(a[None,:],n=size)
      else:
        factory = self.mtype
        def initial(a,size,N=self.N,factory=factory): return spvstack(size*(factory(a,(1,N)),))
      def max_normalise(m):
        x = ones((m.shape[0],1))
        for i,(k,k_) in enumerate(zip(m.indptr[:-1],m.indptr[1:])):
          xi = amax(m.data[k:k_],initial=0.)
          if xi: x[i] = xi; m.data[k:k_] /= xi
        return x
      def product(w1,w2,factory=factory):
        N1,N2 = w1.shape[0],w2.shape[0]
        Z,I,J = zip(*((w1[i1,j1]*w2[i2,j2],N2*i1+i2,N2*j1+j2) for i1,j1 in zip(*w1.nonzero()) for i2,j2 in zip(*w2.nonzero())))
        return factory((Z,(I,J)),shape=2*(N1*N2,))
    else: raise Exception('Should not happen if matrix type has been properly checked')
    self.product,self.initial,self.max_normalise = product,initial,max_normalise

#-------------------------------------------------------------------------------
  @cached_property
  def deterministic(self):
    r"""
Returns whether this automaton is deterministic.
    """
#-------------------------------------------------------------------------------
    r = all([all(sum(w!=0,axis=1)<2) for w in self.W])
    if r and self.N>10 and not issubclass(self.mtype,csra_matrix):
      logger.warning('Deterministic automata with type %s may be inefficient. Use type csra_matrix instead.',self.mtype)
    return r

#-------------------------------------------------------------------------------
  def initialise(self,start:Union[str,int,ndarray],size:int)->lmatrix:
    r"""
Returns an initial state-weight assignment for method :meth:`run` as a log-domain matrix. If *start* is specified as a single state, it is taken to be the one-hot assignment to that state.

:param start: an initial state-weight assignment as a vector of length :math:`N`, the number of states in this automaton
:param size: number :math:`M` of samples
:return: a matrix of shape :math:`M,N`
    """
#-------------------------------------------------------------------------------
    if isinstance(start,str): start = onehot(self.state_names.index(start),self.N)
    elif isinstance(start,int): start = onehot(start,self.N)
    m = self.initial(start,size)
    b = zeros((size,1))
    return lmatrix(m,b,self.max_normalise)

#-------------------------------------------------------------------------------
  def run(self,batch:ndarray,weights:lmatrix):
    r"""
Runs a batch of sequences in log domain. The sequences in the batch all have the same length :math:`L`. The initial weights *weights* are modified inplace.

:param batch: the symbol sequence to evaluate as an :class:`int` matrix of shape :math:`L,M` where :math:`M` is the number of samples
:param weights: state weights as a log-domain matrix of shape :math:`M,N` where :math:`N` is the number of states
    """
#-------------------------------------------------------------------------------
    for csymb in batch:
      for c in unique(csymb): sel=csymb==c; weights[sel] = weights[sel]@self.W[c]

#-------------------------------------------------------------------------------
  def __matmul__(self,other):
#-------------------------------------------------------------------------------
    if not isinstance(other,WFSA): raise ValueError(f'Unsupported types for @: \'{self.__class__}\' and \'{other.__class__}\'')
    if other.mtype != self.mtype: raise ValueError(f'Inconsistent matrix types for @: \'{self.mtype}\' and \'{other.mtype}\'')
    product = self.product
    def intersect():
      for cname,w in zip(self.symb_names,self.W):
        try: cind = other.symb_names.index(cname)
        except ValueError:
          if other.template is not None: yield cname,product(w,other.template)
        else: yield cname,product(w,other.W[cind])
      if self.template is not None:
        for cname,w in zip(other.symb_names,other.W):
          if cname not in self.symb_names: yield cname,product(self.template,w)
    template = None
    if self.template is not None and other.template is not None:
      template = product(self.template,other.template)
    symb_names, W = zip(*intersect())
    state_names = tuple(f'{s1}.{s2}' for s1 in self.state_names for s2 in other.state_names)
    return self.__class__(W,symb_names=symb_names,state_names=state_names,template=template,check=False)

#-------------------------------------------------------------------------------
  def default_state_names(self)->tuple[str,...]: return tuple(str(n) for n in range(self.N))
  def default_symb_names(self)->tuple[str,...]: return tuple(chr(x+97) for x in range(self.n))
#-------------------------------------------------------------------------------
