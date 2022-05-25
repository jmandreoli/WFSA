# File:                 util.py
# Creation date:        2019-11-14
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Utilities for Finite State Automata
#

import logging; logger = logging.getLogger(__name__)
import re,collections,pickle
from contextlib import contextmanager
from pathlib import Path
from numpy import sum, cumsum, ndarray, zeros, nonzero, log, seterr, arange, concatenate, tile, ceil, log10
from numpy import load as npload, save as npsave
from numpy.random import choice
from scipy.sparse import csr_matrix

__all__ = 'Sample', 'FileBatch', 'csra_matrix'

#===============================================================================
class lmatrix:
  r"""
Matrix in log domain. Support only matrix multiplication on the right.

:param m: copy of this matrix where each row has been normalised
:param b: log-coefficients of the normalisation, as a vector of length :math:`N`, the number of rows of this matrix
:type b: :class:`numpy.ndarray`
:param normalise: the function used for the normalisation
:type normalise: :class:`Callable[[numpy.ndarray],numpy.ndarray]`
  """
#===============================================================================
  def __init__(self,m,b,normalise): self.m,self.b,self.normalise = m,b,normalise
  def __getitem__(self,s): return lmatrix(self.m[s],self.b[s],self.normalise)
  def __setitem__(self,s,v): self.m[s] = v.m; self.b[s] = v.b
  def __matmul__(self,other):
    m = self.m@other
    x = self.normalise(m)
    return lmatrix(m,self.b+log(x),self.normalise)
  def logvalue(self):
    r"""
:returns: the logvalues of this matrix
:rtype: :class:`numpy.ndarray`
    """
    with seterr_c(divide='ignore'): return log(toarray(self.m))+self.b
  def __repr__(self): return f'{self.m}*exp{self.b}'

#===============================================================================
class csra_matrix (csr_matrix):
  r"""
Instances of this class are matrices with at most one non null value per row. This implementation just refines class :class:`scipy.sparse.csr_matrix` to more efficiently (?) implement matrix multiplication, lookup and a limited form of inplace change. Ideally, it should be robustified and implemented in C++.
  """
#===============================================================================
  @staticmethod
  def initial(*a,n=1,**ka):
    r"""
Same arguments as the :class:`scipy.sparse.csr_matrix` constructor, with additional keyword argument *n*. Returns a :class:`csra_matrix` instance (when possible) stacked *n* times on itself.
    """
    def ini(A):
      for k,k_ in zip(A.indptr[:-1],A.indptr[1:]):
        if k_ == k: yield 0,0.
        else:
          if k_!=k+1: raise ValueError('Constructor \'initial\' works only with onehot matrices')
          yield A.indices[k],A.data[k]
    A = csr_matrix(*a,**ka)
    indices,data = zip(*ini(A))
    indices,data = tile(indices,n),tile(data,n)
    n, = indices.shape
    return csra_matrix((data,indices,arange(n+1)),(n,A.shape[1]))
  def neutralise(self):
    return csr_matrix((self.data,self.indices,self.indptr),self.shape)
  def __matmul__(self,other):
    if isinstance(other,csra_matrix):
      if self.shape[1] != other.shape[0]: raise ValueError(f'Inconsistent shapes for matrix multiplication: {self.shape} {other.shape}')
      return csra_matrix((self.data*other.data[self.indices],other.indices[self.indices],self.indptr),(self.shape[0],other.shape[1]))
    else: # looses csra property...
      return self.neutralise().__matmul__(other)
  def __getitem__(self,key):
    key0,key1 = key if isinstance(key,tuple) else (key,slice(None))
    if (isinstance(key0,slice) or (isinstance(key0,ndarray) and len(key0.shape)==1 and key0.dtype==bool)) and (isinstance(key1,slice) or (isinstance(key1,ndarray) and len(key1.shape)==1 and key1.dtype==bool)):
      data,indices = self.data[key0],self.indices[key0]
      n2 = self.shape[1]
      if isinstance(key1,slice):
        if key1 != slice(None): raise NotImplementedError('To be done') # XXXX
      else:
        sel = ~key1[indices]
        newpos = cumsum(key1)-1
        indices = newpos[indices]; indices[sel] = 0
        data = data.copy(); data[sel] = 0.
        n2 = newpos[-1]+1
      n, = indices.shape
      return csra_matrix((data,indices,self.indptr[:n+1]),(n,n2))
    else: # looses csra property...
      return self.neutralise().__getitem__(key)
  def __setitem__(self,key,val): # works only in a very specific case
    if not (isinstance(val,csra_matrix)): raise TypeError(f'Unsupported type for setitem: {self.__class__} and {val.__class__}')
    if not(isinstance(key,ndarray) and key.dtype==bool): raise TypeError(f'Unsupported key type for setitem (only boolean masks): {key.__class__}')
    self.indices[key],self.data[key] = val.indices,val.data
  def __add__(self,other): return self.neutralise().__add__(other)
  def __radd__(self,other): return self.neutralise().__radd__(other)
  def __sub__(self,other): return self.neutralise().__sub__(other)
  def __rsub__(self,other): return self.neutralise().__rsub__(other)

#===============================================================================
class Sample (collections.abc.Mapping):
  r"""
An instance of this class is a sample of variable-length sequences of ints presented as uniform length batches of diminishing size. For example, the sample (of size 2)::

   2 2 5 3 1
   1 3 4 2 4 1 5 3

could be represented as 3 batches of size 2,2,1 and length, 3,3,2 (\* is a padding value)::

   1 3 4  |  2 4 1  |  5 3
   2 2 5  |  3 1 *

(actually, the disposition is vertical, i.e. length is first dim and size is second).

:param size: number of items in the sample
:type size: :class:`int`
:param store: function used to store a batch, returning a stored batch
:type store: :class:`Callable[[numpy.ndarray],Any]`
  """
#===============================================================================
  def __init__(self,size,store=None):
    if store is None: store = (lambda batch:batch)
    self.size,self.content,self.closed,self.store = size,[],False,store

#-------------------------------------------------------------------------------
  def append(self,batch,sel=None,nsize=0):
    r"""
Appends a batch to this sample.

:param batch: the batch to add, as an :class:`int` array of shape :math:`L,M`, where :math:`L` is the common length of the sequences of the batch and :math:`M` is the number of sequences in the batch
:type batch: :class:`numpy.ndarray`
:param sel: the selection (:class:`bool` vector of length :math:`M`) of the current batch items which have a continuation in the next batch
:type sel: :class:`numpy.ndarray`
:param nsize: number of items in the current batch which have a continuation in the next batch
:type nsize: :class:`int`
    """
#-------------------------------------------------------------------------------
    if self.closed: raise ValueError('Cannot append to closed sample')
    size = batch.shape[1]
    #assert size<=self.size
    if nsize:
      if sel is None: ind = arange(size) # cannot use slice trick
      else:
        ind = concatenate((nonzero(sel)[0],nonzero(~sel)[0]))
        for b_,n_,i in self.content: i[:size] = i[ind]
    else: ind = slice(None) # slice trick: only on the last batch
    self.content.append((self.store(batch),nsize,ind))

#-------------------------------------------------------------------------------
  def close(self):
    r"""
Closes this sample. Reorders all the batches so that the items of a batch having a continuation in the next are regrouped before the others.
    """
#-------------------------------------------------------------------------------
    def close():
      for sbatch,nsize,ind in self.content:
        sbatch[...] = sbatch[:,ind]
        yield sbatch,nsize
    if not self.closed:
      self.content = tuple(close())
      self.closed = True
    return self

#-------------------------------------------------------------------------------
  def pprint(self,symb_names=None,pick=10,translate={},file=None,mag=lambda x: int(ceil(log10(x)))):
    r"""
Prints a subsample of this sample obtained by picking *pick* items at random.

:param symb_names: a mapping from the :class:`int` values in the sample and unicode characters
:type symb_names: :class:`Iterable[str]`
:param pick: size of the sub-sample
:type pick: :class:`int`
:param file: passed to :func:`print`
    """
#-------------------------------------------------------------------------------
    if symb_names is None: symb_names=[chr(x) for x in range(97,123)]
    mk = mag(self.size)
    mv = mag(sum([batch.shape[0] for batch,o in self.content]))
    for k in choice(self.size,pick,replace=False):
      s = ''.join(symb_names[c] for c in self[k]).translate(translate)
      print(str(k).rjust(mk),' ',str(len(s)).rjust(mv),' ',s,file=file)

  def __len__(self): return self.size
  def __iter__(self): yield from range(self.size)
  def __getitem__(self,k):
    if not self.closed: raise ValueError('Cannot get item from open sample')
    def getitem(k):
      for sbatch,nsize in self.content:
        yield sbatch[:,k]
        if k>=nsize: break
    return concatenate(list(getitem(k)))

#===============================================================================
class FileBatch:
  r"""
Instances of this class provide a file based store function for sample batches.

:param path: a path to a folder for batch storage
:type path: :class:`Union[str,pathlib.Path]`
  """
#===============================================================================
  def __init__(self,path,clear=False):
    if isinstance(path,str): path = Path(path)
    elif not isinstance(path,Path): raise TypeError('Argument must be of type Union[str,Path]')
    self.path = path
    self.base = path/'batch_'
    self.getpath = lambda n,path=path: path/f'batch_{n:03x}.npy'
    if self.base.exists():
      if clear: self.clear()
    else:
      path.mkdir(parents=True)
      with self.base.open('wb') as v: pickle.dump(0,v)
#-------------------------------------------------------------------------------
  def clear(self):
    r"""
Clears the batch folder.
    """
#-------------------------------------------------------------------------------
    for f in list(self.path.glob('batch_*')): f.unlink()
    with self.base.open('wb') as v: pickle.dump(0,v)
#-------------------------------------------------------------------------------
  def store(self,batch):
    r"""
Stores a batch on file and returns a stored batch. This method can be passed as *store* argument to create an open :class:`Sample` instance.
    """
#-------------------------------------------------------------------------------
    with self.base.open('rb+') as u:
      n = int(pickle.load(u)); u.seek(0)
      pickle.dump(n+1,u); u.truncate()
    p = self.getpath(n)
    npsave(p,batch)
    return npload(p,mmap_mode='r+')
#-------------------------------------------------------------------------------
  def toSample(self):
    r"""
Returns a closed :class:`Sample` instance.
    """
#-------------------------------------------------------------------------------
    with self.base.open('rb') as u: n = int(pickle.load(u))
    L = [npload(self.getpath(k),mmap_mode='r') for k in range(n)]
    sample = Sample(L[0].shape[1],None)
    content = [(r,r_.shape[1]) for r,r_ in zip(L[:-1],L[1:])]
    content.append((L[-1],0))
    sample.content = tuple(content)
    sample.closed = True
    return sample

#===============================================================================
class WFSABrowsePlugin:
  r"""
Some utilities to browse automata. Kept separate as a plugin for readability.
  """
#===============================================================================

  maxN = 50

#-------------------------------------------------------------------------------
  def as_html(self,
    dfstyle='border:thin solid black;text-align:center;',
    hdstyle='background-color:blue;color: white;',
    vhdstyle='writing-mode:vertical-lr;min-width:.5cm;',
    cstyle='font-size:x-small'):
    r"""
Returns an html table of the transitions. Use :func:`IPython.display.display` for a nice display in a notebook.
    """
#-------------------------------------------------------------------------------
    from lxml.html import tostring
    from lxml.html.builder import TABLE,TR,TD,B
    def rows():
      def hdr():
        yield info
        for s,n in enumerate(self.state_names): yield B(n),dict(style=dfstyle+hdstyle+empfinal(s)+vhdstyle)
        if self.template is not None:
          yield '🛈',dict(title='template',style=dfstyle+'border-left:thick solid black;')
          for s,n in enumerate(self.state_names): yield B(n),dict(style=dfstyle+hdstyle+empfinal(s)+vhdstyle)
      def row(s,n):
        yield B(n),dict(style=dfstyle+hdstyle+empfinal(s))
        for s2,x in enumerate(toarray(wbar[s]).squeeze()): yield (f'{x:.2}' if x else ''),dict(title=detail(x,*((self.symb_names[c],w[s,s2]) for c,w in enumerate(self.W))),style=dfstyle+cstyle)
        if self.template is not None:
          yield B(n),dict(style=dfstyle+hdstyle+empfinal(s)+'border-left:thick solid black;')
          for s2,x in enumerate(toarray(self.template[s]).squeeze()): yield (f'{x:.2}' if x else ''),dict(title=detail(x),style=dfstyle+cstyle)
      yield hdr()
      for s,n in enumerate(self.state_names): yield row(s,n)
    def detail(x,*l):
      return f'{x:.5}\n'+'\n'.join(f'{c} {v:.5}' for c,v in l if v)
    def empfinal(s): return 'background-color: darkblue;' if hasattr(self,'final') and s==self.final else ''
    def td(x,atts=None):
      if atts is None: atts = {}
      atts.setdefault('style',dfstyle)
      return TD(x,**atts)
    info = '🛈',dict(title=f'{self.N} states, {self.n} symbols, {sum([sum(w!=0) for w in self.W])} transitions\nmtype: {self.mtype}',style=dfstyle+'background-color:white;color:blue;')
    if self.N<=self.maxN:
      wbar = 0.
      for w in self.W: wbar += w
      html = TABLE(*(TR(*(td(*x) for x in row)) for row in rows()))
    else: html = TABLE(TR(td(*info)))
    return tostring(html,encoding=str)

#-------------------------------------------------------------------------------
  graph_ = None
  @property
  def graph(self):
    r"""
Returns a networkx representation of this automaton.
    """
#-------------------------------------------------------------------------------
    from networkx import DiGraph
    g = self.graph_
    if g is None:
      if self.N>self.maxN: raise ValueError('Model too large; graph cannot be built')
      self.graph_ = g = DiGraph()
      for i,n in enumerate(self.state_names): g.add_node(i,name=n,transition=False)
      i = -1
      D = self.symb_names
      for c,w in enumerate(self.W):
        for sfrom,sto in zip(*nonzero(w)):
          v = w[sfrom,sto]
          g.add_node(i,name=D[c],transition=True)
          g.add_edge(sfrom,i)
          g.add_edge(i,sto,weight=v)
          i -= 1
    return g

#-------------------------------------------------------------------------------
  def draw(self,tune={},ax=None,**ka):
    r"""
Draws this automaton as a network (not so great)
    """
#-------------------------------------------------------------------------------
    from networkx import bipartite_layout, spring_layout, rescale_layout, draw_networkx, draw_networkx_edge_labels
    from matplotlib.pyplot import figure
    if ax is None: ax = figure(**ka).add_subplot(1,1,1)
    elif ka: raise ValueError('Keyword arguments allowed only when ax is not given')
    g = self.graph
    k = tune.get('k',.5); K = tune.get('K',5.)
    pos = dict((i,((K/2*(next(g.successors(i))+next(g.predecessors(i))),-1) if n['transition'] else (K*i,0))) for i,n in g.nodes.items())
    pos = spring_layout(
      g,k,
      pos=pos,
      fixed=[i for i,n in g.nodes.items() if not n['transition']],
      weight=None,
    )
    draw_networkx(
      g,pos,
      node_color=[('m' if n['transition'] else 'c') for n in g.nodes.values()],
      edge_color=[('k' if 'weight' in e else 'c') for e in g.edges.values()],
      labels=dict((i,n['name']) for i,n in g.nodes.items()),
      ax=ax,
    )
    draw_networkx_edge_labels(
      g,pos,label_pos=.6,
      edge_labels=dict(((i,j),'{:.2f}'.format(e['weight'])) for (i,j),e in g.edges.items() if 'weight' in e),
      font_size=8,
      ax=ax,
    )
    return g

#-------------------------------------------------------------------------------
  @classmethod
  def from_str(cls,descr,sparse=False,sep='\n',pat=re.compile(r'(\w+|)->(\w+|)\s+(\S|)\s+([0-9.]+)')):
    r"""
Returns a :class:`.WFSA` instance described by parameter *descr*, which must be a string in a simple syntax. This is a class method and the created instance is of the class from which it is invoked. For example, a trivial probabilistic automaton for strings of geometric lengths over a singleton alphabet can be specified as:

.. code:: python

   PFSA.from_str('''
     O->O a .3
     O->F . .7
     F->F . 1.
   ''')
    """
#-------------------------------------------------------------------------------
    def add(x,D):
      if not x in D: D[x] = len(D)
    L = [pat.fullmatch(x.strip()).groups() for x in descr.strip().split(sep)]
    Dstate,Dsymb = {},{}
    for x in L: add(x[0],Dstate); add(x[2],Dsymb)
    notr = [x[1] for x in L if x[1] not in Dstate]
    if notr: raise ValueError(f'Some states have no transition: {notr}')
    N = len(Dstate)
    W = zeros((len(Dsymb),N,N))
    for sfrom,sto,symb,w in L: W[Dsymb[symb],Dstate[sfrom],Dstate[sto]] = float(w)
    if sparse: W = [csra_matrix.initial(w) for w in W]
    return cls(tuple(W),state_names=tuple(Dstate),symb_names=tuple(Dsymb))

#===============================================================================
# Miscelanous
#===============================================================================

@contextmanager
def seterr_c(**ka):
  errconfig = seterr(**ka)
  try: yield
  finally: seterr(**errconfig)

def onehot(n,N,val=1.,**ka):
  a = zeros(N,**ka)
  a[n] = val
  return a

def toarray(a):
  return a.toarray() if hasattr(a,'toarray') else a
