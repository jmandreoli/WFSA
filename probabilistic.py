# File:                 probabilistic.py
# Creation date:        2019-11-14
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Implementation of Probabilistic Finite State Automata
#
r"""
:mod:`WFSA.probabilistic` --- Probabilistic Finite State Automata
=================================================================

This module provides core functionalities for Probabilistic FSA.

Available types and functions
-----------------------------
"""

import logging
logger = logging.getLogger(__name__)
import collections
from numpy import ndarray, all, allclose, sum, array, eye, transpose, zeros, bincount, empty, nonzero, log, exp, unique, argmax, average, ceil, stack, amin, amax
from numpy.random import choice
from numpy.linalg import solve
from scipy.special import xlogy
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve
from .core import WFSA
from .util import NameMap, Sample, onehot

#===============================================================================
class PFSA (WFSA):
  r"""
An instance of this class is a Probabilistic Finite State Automata. It is also an instance of WFSA. It must have a unique singleton attractor, which is taken to be the final state. There must be a unique transition from the final state, and it emits the stop symbol. States from which the final state is not reachable are pruned. All the parameters of WFSA are honored, as well as

:param stop: the index or name of the stop symbol (optional)
:type stop: :class:`Union[NoneType,str,int]`
:param check2: whether to make PFSA specific verifications at initialisation
:type check2: :class:`bool`

Attributes (in addition to those of :class:`.WFSA`):

.. attribute:: stop

   The index of the stop symbol, guessed if not provided or :const:`None`

.. attribute:: Wbar

   The global transition matrix (sum of the transition matrices of all the symbols)

.. attribute:: final

   The index of the final state, inferred from the stop symbol or guessed together with it

.. attribute:: nonfinal

   A filter (1D :class:`numpy.ndarray` of type :class:`bool`) to select all the states except the final one

.. attribute:: normaliser

   The normaliser to apply to an initial state weight vector before turning it into a distribution

Methods:
  """
#===============================================================================

#-------------------------------------------------------------------------------
  def init_struct(self,*a,stop=None,check2=True,**ka):
#-------------------------------------------------------------------------------
    super().init_struct(*a,**ka)
    # matrix type specific functions needed here
    if issubclass(self.mtype,ndarray):
      def normalise(a,z):
        for z_,a_ in zip(z,a): a_ *= z_/z
      solve_,eye_ = solve,eye
    else:
      def normalise(a,z):
        for i,(k,k_) in enumerate(zip(a.indptr[:-1],a.indptr[1:])):
          a.data[k:k_] *= z[a.indices[k:k_]]/z[i]
      solve_,eye_ = spsolve,speye
    # computation of the global transition matrix
    wbar = 0.
    for w in self.W: wbar += w
    # determination of stop symbol and final state
    if stop is None:
      stopc = []
      for c,w in enumerate(self.W):
        a = sum(w,axis=0)!=0
        if sum(a)==1:
          f = argmax(a)
          if not sum(wbar[f]!=w[f]): stopc.append((c,f))
      if check2:
        if len(stopc)!=1: raise ValueError('Stop symbol not found or ambiguous: {}'.format(stopc))
      (stop,final), = stopc
      logger.info('Inferred stop symbol and final state: %s %s',self.symb_names[stop],self.state_names[final])
    else:
      if isinstance(stop,str): stop = self.symb_names._[stop]
      a = sum(self.W[stop],axis=0)!=0
      final = argmax(a)
      if check2:
        if sum(a)!=1: raise ValueError('Final state not found or ambiguous')
        if sum(wbar[final]!=self.W[stop][final]): raise ValueError('Non stop symbol transition found from final state')
      logger.info('Inferred final state: %s',self.state_names[final])
    # pruning of states which cannot reach the final state
    wbar_b = wbar!=0; wbar_b += eye_(self.N,dtype=bool)
    q = onehot(final,self.N,dtype=bool)
    for _ in range(self.N): q[...] = wbar_b@q
    if not all(q):
      # if actual pruning, recompute everything concerning states
      N = sum(q)
      logger.info('Pruning unreachable states: %s/%s',self.N-N,self.N)
      self.N = N
      self.W = tuple(w[q,:][:,q] for w in self.W)
      wbar = wbar[q,:][:,q]
      self.state_names = NameMap(array(self.state_names)[q])
      final = sum(q[:final])
    # set up of pfsa specific attributes
    self.stop = stop
    self.Wbar = wbar
    self.final = final
    self.nonfinal = nonfinal = ~onehot(final,self.N,dtype=bool)
    # normalisation of weights
    self.normaliser = None
    if not allclose(sum(wbar,axis=1),1.):
      wnf = wbar[nonfinal]
      self.normaliser = z = empty(self.N)
      z[-1] = 1.
      z[:-1] = solve_(eye_(self.N-1)-wnf[:,nonfinal],wnf[:,final])
      if not all(z>0): raise ValueError('Non positive normalisation coefficients')
      logger.info('Normaliser min: %s max: %s',amin(z),amax(z))
      normalise(wbar,z)
      for w in self.W: normalise(w,z)

#-------------------------------------------------------------------------------
  def init_type(self):
#-------------------------------------------------------------------------------
    super().init_type()
    # Special methods dependent on matrix type
    wbar = self.Wbar
    if issubclass(self.mtype,ndarray):
      sample_next = lambda s,count,N=self.N,wbar=wbar: choice(N,count,p=wbar[s])
      wnorm = stack(self.W).transpose(1,2,0)
      sel = wbar!=0
      wnorm[sel,:] /= wbar[sel,None]
      sample_symbol = lambda s,s2,count,n=self.n,wnorm=wnorm: choice(n,count,p=wnorm[s,s2])
      eye_,solve_ = eye,solve
    elif issubclass(self.mtype,csr_matrix):
      wbar = self.Wbar
      def sample_next(s,count,wbar=wbar): a = wbar[s]; return choice(a.indices,count,p=a.data)
      wnorm = {}
      for i,(k,k_) in enumerate(zip(wbar.indptr[:-1],wbar.indptr[1:])):
        for j,v in zip(wbar.indices[k:k_],wbar.data[k:k_]):
          if v: wnorm[i,j] = array([w[i,j]/v for w in self.W])
      sample_symbol = lambda s,s2,count,n=self.n,wnorm=wnorm: choice(n,count,p=wnorm[s,s2])
      eye_,solve_ = speye,spsolve
    self.sample_next,self.sample_symbol,self.eye,self.solve = sample_next,sample_symbol,eye_,solve_

#-------------------------------------------------------------------------------
  def logproba(self,sample,start):
    r"""
Evaluates the probability of a set of strings in log domain for a given initial state distribution. If *start* is specified as a single state, it is taken to be the one-hot assignment to that state.

:param sample: the :math:`M` strings to evaluate
:type sample: :class:`.Sample`
:param start: an initial state distribution as a vector of length :math:`N`, the number of states in this automaton
:type start: :class:`Union[str,int,numpy.ndarray]`
:return: output log-probabilities as a vector of length :math:`M`
:rtype: :class:`numpy.ndarray`
    """
#-------------------------------------------------------------------------------
    probs = probs_ = self.initialise(start,sample.size)
    for batch,nsize in sample.content:
      self.run(batch,probs_)
      probs_ = probs_[:nsize]
    return (probs@self.W[self.stop][:,[self.final]]).logvalue()[:,0]

#-------------------------------------------------------------------------------
  expected_length_ = None
  def expected_length(self,start=None):
    r"""
Returns the expected length of strings under the distribution of this automaton with initial state given by parameter *start*, either a state's name or index. If *start* is omitted, returns a table of values for each initial state.

:parameter start: initial state
:type start: :class:`Union[str,int,NoneType]`
    """
#-------------------------------------------------------------------------------
    r = self.expected_length_
    if r is None:
      W = self.Wbar-self.W[self.stop]
      r = self.expected_length_ = self.solve(self.eye(self.N)-W,sum(W,axis=1))
    return r if start is None else r[self.state_names._[start] if isinstance(start,str) else start]

#-------------------------------------------------------------------------------
  entropy_ = None
  def entropy(self,start=None):
    r"""
Returns the entropy of the distribution of this model with initial state given by parameter *start*, either a state's name or index. If *start* is omitted, returns a table of values for each initial state. Note: works only if the transition associated with each symbol is deterministic.

:parameter start: initial state
:type start: :class:`Union[str,int,NoneType]`
    """
#-------------------------------------------------------------------------------
    def ent(w): return -xlogy(w,w)
    r = self.entropy_
    if r is None:
      if not self.deterministic: raise NotImplementedError('Entropy available only for deterministic automata')
      r = self.entropy_ = zeros(self.N)
      u = sum(ent(sum(w,axis=1)) for w in self.W)[self.nonfinal]
      r[self.nonfinal] = self.solve((self.eye(self.N)-self.Wbar)[self.nonfinal,:][:,self.nonfinal],u)
    return r if start is None else r[self.state_names._[start] if isinstance(start,str) else start]

#-------------------------------------------------------------------------------
  def sample(self,size,start,store=None):
    r"""
Returns a sample of this automaton. If *start* is specified as a single state, it is taken to be the one-hot assignment to that state.

:param size: number :math:`M` of samples required
:type size: :class:`int`
:param start: input state distribution as a vector of length :math:`N`, the number of states of this automaton
:type start: :class:`Union[str,int,numpy.ndarray]`
:return: sample as an array of shape :math:`L,M` where :math:`L` the max length of the sample (each string is terminated by a non empty suffix of stop symbols).
    """
#-------------------------------------------------------------------------------
    # scount: previous state counts
    scount = None
    if isinstance(start,str): start = self.state_names._[start]
    elif not isinstance(start,int):
      start = choice(self.N,size,p=start)
      scount = bincount(start,minlength=self.N)
    if scount is None: scount = onehot(start,self.N,size,dtype=int)
    # pstate, cstate: previous/current states
    pstate,cstate = empty((2,size),dtype=int)
    pstate[...] = start
    # content: stack of generations
    sample = Sample(size,store)
    while True:
      # trick here: generate the next state first then generate the symbol
      batchlen = int(ceil(average(self.expected_length(),weights=scount)))
      batch = empty((batchlen,size),dtype=int)
      # indices of non terminated particles
      for nstep,csymb in enumerate(batch):
        for s,n in [(s,scount[s]) for s in nonzero(scount)[0]]:
          # iterator must be listed as scount later changes
          sel = pstate==s
          if s==self.final:
            if n==size:
              sample.append(batch[:nstep])
              return sample.close()
            else:
              cstate[sel] = self.final
              csymb[sel] = self.stop
          else:
            isel = nonzero(sel)[0]
            cstate[sel] = cstate2 = self.sample_next(s,n)
            ccount = bincount(cstate2)
            scount[s] -= n
            scount[:len(ccount)] += ccount
            for s2 in nonzero(ccount)[0]:
              csymb[isel[cstate2==s2]] = self.sample_symbol(s,s2,ccount[s2])
        pstate,cstate = cstate,pstate
      sel = pstate!=self.final
      size_ = sum(sel)
      if size_==0:
        sample.append(batch)
        return sample.close()
      if size_<size: pstate,cstate = pstate[sel],cstate[:size_]
      else: sel = None
      scount[self.final] = 0
      sample.append(batch,sel,size_)
      size = size_

#===============================================================================
def toPFSA(self,stop=None,check2=True): # this is a WFSA method, not PFSA!
  r"""
Returns a copy of this automaton as a :class:`.PFSA`.

:param stop,check2: same role as in the :class:`.PFSA` constructor
  """
#===============================================================================
  return PFSA([w.copy() for w in self.W],state_names=self.state_names[:],symb_names=self.symb_names[:],check=False,stop=stop,check2=check2)
WFSA.toPFSA = toPFSA
