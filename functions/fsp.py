from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Optional, Iterable
from itertools import product

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply





# BASIC SETUP -----

State = Tuple[int, ...]  # e.g., (X0, X1, C00, C10, C11)

@dataclass(frozen=True)
class Reaction:
    nu: np.ndarray                               # shape (D,), integer stoichiometry change
    rate_fn: Callable[[np.ndarray], float]       # a(x): R^D -> R_+


# ===================



# Solve CME via sparse matrix exponentiation (recall: dp/dt = H p  =>     p(t) = exp(H t) p(0)  )
def solve_cme_fsp(Q: csc_matrix, p0: np.ndarray, t_init, t_final, num_t):
    """
    Compute p(t) for linearly spaced t from t_init to t_final via expm_multiply.
    Q required to be a sparse matrix (https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)
    Returns array with shape (num_t, Q.shape[0])
    """
    return expm_multiply(Q, p0, start=t_init, stop=t_final, num=num_t, endpoint=True)


# BUILDING GENERATOR ----------


def enumerate_states_box(bounds: List[Tuple[int,int]],
                         constraint: Callable[[State], bool] = lambda s: True
                         ) -> List[State]:
    """
    Rectangular box enumeration with an optional constraint filter.
    bounds: [(min0,max0), ...] inclusive bounds per species
    """
    grids = [range(lo, hi+1) for (lo, hi) in bounds]
    states = []
    for s in product(*grids):
        if constraint(s):
            states.append(tuple(s))
    return states



# has weird broadcasting error for one species models; works otherwise
def build_generator_fsp(states: List[State], reactions: List[Reaction], add_sink: bool = True) -> csc_matrix:
    """
    Build sparse generator Q over 'states', optionally with a sink as the last index.
    Columns sum to zero (i.e., Q has columns as source states).
    """

    index: Dict[State, int] = {s:i for i,s in enumerate(states)}
    S = len(states)
    sink_idx = S if add_sink else None

    rows, cols, data = [], [], []

    for j, s in enumerate(states):
        x = np.array(s, dtype=int)
        a_sum = 0.0
        for rxn in reactions:
            a = rxn.rate_fn(x)
            if a <= 0.0:
                continue
            y = tuple((x + rxn.nu).tolist())
            a_sum += a
            if y in index:
                i = index[y]
            elif add_sink:
                i = sink_idx
            else:
                # drop flux that leaves the truncation (not recommended)
                continue
            # off-diagonal entry: flux from j -> i
            rows.append(i); cols.append(j); data.append(a)
        # diagonal (loss)
        rows.append(j); cols.append(j); data.append(-a_sum)

    n = S + (1 if add_sink else 0)
    Q = csc_matrix((data, (rows, cols)), shape=(n, n))
    return Q, index

# =========================


# MARGINALIZATION ---------
# Note: some of this machinery wasn't used in the final version. 
# Part of it relates to the general case of a (countably) infinite state space. If you try to handle this,
# need to add a 'sink' state which complicates certain computations like marginalization.


def _strip_sink(P: np.ndarray, states: List[Tuple[int, ...]], include_sink: bool):
    S = len(states)
    if P.shape[1] == S:
        return P, S, None if not include_sink else np.zeros((P.shape[0], 1), dtype=P.dtype)
    elif P.shape[1] == S + 1:
        sink = P[:, -1:]
        return (P[:, :S], S, sink) if include_sink else (P[:, :S], S, None)
    else:
        raise ValueError("P second dimension must be |states| or |states|+1 (sink).")

def joint_marginal_over_vars(
    P: np.ndarray,
    states: Iterable[Tuple[int, ...]],
    keep_vars: List[int],
    include_sink: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    Pm, S, sink = _strip_sink(P, list(states), include_sink)
    T = Pm.shape[0]
    states_arr = np.array(list(states), dtype=int)
    kept = states_arr[:, keep_vars]
    K = kept.shape[1]

    maxs = kept.max(axis=0)
    axes_values = [np.arange(m+1, dtype=int) for m in maxs]
    out_shape = (T,) + tuple(m+1 for m in maxs)
    M = np.zeros(out_shape, dtype=Pm.dtype)

    if K == 1:
        ind0 = kept[:, 0]
        for t in range(T):
            np.add.at(M[t], ind0, Pm[t, :S])
    elif K == 2:
        ind0, ind1 = kept[:, 0], kept[:, 1]
        for t in range(T):
            np.add.at(M[t], (ind0, ind1), Pm[t, :S])
    elif K == 3:
        ind0, ind1, ind2 = kept[:, 0], kept[:, 1], kept[:, 2]
        for t in range(T):
            np.add.at(M[t], (ind0, ind1, ind2), Pm[t, :S])
    else:
        idx_tuples = [kept[:, k] for k in range(K)]
        for t in range(T):
            for s in range(S):
                M[(t,)+tuple(kt[s] for kt in idx_tuples)] += Pm[t, s]

    return M, axes_values, sink

def marginal_over_function(
    P: np.ndarray,
    states: Iterable[Tuple[int, ...]],
    key_fn: Callable[[Tuple[int, ...]], int],
    keys: Optional[np.ndarray] = None,
    include_sink: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    Pm, S, sink = _strip_sink(P, list(states), include_sink)
    T = Pm.shape[0]
    states_list = list(states)
    k_vals = np.array([key_fn(s) for s in states_list], dtype=int)
    if keys is None:
        keys = np.unique(k_vals)
    key_to_pos = {k:i for i,k in enumerate(keys)}
    M = np.zeros((T, keys.size), dtype=Pm.dtype)
    pos = np.array([key_to_pos[k] for k in k_vals], dtype=int)
    for t in range(T):
        np.add.at(M[t], pos, Pm[t, :S])
    return M, keys, sink






# ==================


# Propensities and stoichiometry vectors for various models



def prop_crick(x, params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    # x: (num_samples, D)
    
    # species:
    # C00, C10, C11
    
    prop = np.array([
                    2*eps*x[:,0],              # C00 -> C10 (2 eps)
                    (eps+k)*x[:,1],            # C10 -> C11 (eps + kcat)
                    2*gamma*x[:,2],            # C11 -> C10 (2 delta)
                    gamma*x[:,1],              # C10 -> C00 (delta)
        ]).T
    
    return prop

def stoich_crick():
    # species:
    # C00, C10, C11
    
    stoich = np.array([ 
                    [-1,+1,0],     # C00 -> C10 (2 eps)
                    [0,-1,+1],     # C10 -> C11 (eps + kcat)
                    [0,+1,-1],     # C11 -> C10 (2 delta)
                    [+1,-1,0],     # C10 -> C00 (delta)
                    ])    
    return stoich


def prop_crick_variance(x, params):
    alpha, beta = params['alpha'], params['beta']
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    
    # species:
    # C00, C10, C11, Z
    prop = np.array([
                    2*eps*x[:,0],              # C00 -> C10 (2 eps)
                    (eps+k)*x[:,1],            # C10 -> C11 (eps + kcat)
                    2*gamma*x[:,2],            # C11 -> C10 (2 delta)
                    gamma*x[:,1],              # C10 -> C00 (delta)
                    beta*x[:,0],               # C00 -> Z 
                    beta*x[:,1],               # C10 -> Z 
                    beta*x[:,2],               # C11 -> Z
                    alpha*x[:,3],              # Z -> C00
        ]).T
    
    return prop


def stoich_crick_variance():    
    # species:
    # C00, C10, C11, Z
    
    stoich = np.array([ 
                    [-1,+1,0,0],     # C00 -> C10 (2 eps)
                    [0,-1,+1,0],     # C10 -> C11 (eps + kcat)
                    [0,+1,-1,0],     # C11 -> C10 (2 delta)
                    [+1,-1,0,0],     # C10 -> C00 (delta)
                    [-1,0,0,+1],     # C00 -> Z 
                    [0,-1,0,+1],     # C10 -> Z 
                    [0,0,-1,+1],     # C11 -> Z
                    [+1,0,0,-1],     # Z -> C00
                    ])    
    return stoich


def prop_crick_binding(x, params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    bf, br = params['bf'], params['br']
    
    prop = np.array([
                    eps*x[:,0],                # X0 -> X1
                    gamma*x[:,1],              # X1 -> X0
                    bf*0.5*x[:,0]*(x[:,0]-1),  # X0 + X0 -> C00
                    bf*x[:,0]*x[:,1],          # X0 + X1 -> C10
                    bf*0.5*x[:,1]*(x[:,1]-1),  # X1 + X1 -> C11
                    br*x[:,2],                 # C00 -> X0 + X0
                    br*x[:,3],                 # C10 -> X0 + X0
                    br*x[:,4],                 # C11 -> X1 + X1
                    2*eps*x[:,2],              # C00 -> C10 (2 eps)
                    (eps+k)*x[:,3],            # C10 -> C11 (eps + kcat)
                    2*gamma*x[:,4],            # C11 -> C10 (2 delta)
                    gamma*x[:,3],              # C10 -> C00 (delta)
        ]).T
    return prop


def stoich_crick_binding():
    # species:
    # X0, X1, C00, C10, C11
    
    stoich = np.array([ 
                    [-1,+1,0,0,0],    # X0 -> X1
                    [+1,-1,0,0,0],    # X1 -> X0
                    [-2,0,+1,0,0],    # X0 + X0 -> C00
                    [-1,-1,0,+1,0],    # X0 + X1 -> C10
                    [0,-2,0,0,+1],    # X1 + X1 -> C11
                    [+2,0,-1,0,0],    # C00 -> X0 + X0
                    [+1,+1,0,-1,0],    # C10 -> X0 + X0
                    [0,+2,0,0,-1],    # C11 -> X1 + X1
                    [0,0,-1,+1,0],    # C00 -> C10 (2 eps)
                    [0,0,0,-1,+1],    # C10 -> C11 (eps + kcat)
                    [0,0,0,+1,-1],    # C11 -> C10 (2 delta)
                    [0,0,+1,-1,0],    # C10 -> C00 (delta)
                    ])
    return stoich





def prop_crick_full(x, params):
    alpha, beta = params['alpha'], params['beta']
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    bf, br = params['bf'], params['br']
    
    # species:
    # X0, X1, C00, C10, C11
    
    prop = np.array([
                    eps*x[:,0],                # X0 -> X1
                    gamma*x[:,1],              # X1 -> X0
                    bf*0.5*x[:,0]*(x[:,0]-1),  # X0 + X0 -> C00
                    bf*x[:,0]*x[:,1],          # X0 + X1 -> C10
                    bf*0.5*x[:,1]*(x[:,1]-1),  # X1 + X1 -> C11
                    br*x[:,2],                 # C00 -> X0 + X0
                    br*x[:,3],                 # C10 -> X0 + X0
                    br*x[:,4],                 # C11 -> X1 + X1
                    2*eps*x[:,2],              # C00 -> C10 (2 eps)
                    (eps+k)*x[:,3],            # C10 -> C11 (eps + kcat)
                    2*gamma*x[:,4],            # C11 -> C10 (2 delta)
                    gamma*x[:,3],              # C10 -> C00 (delta),
                    beta*x[:,0],               # X0 -> Z
                    beta*x[:,1],               # X1 -> Z
                    alpha*x[:,5],              # Z -> X0
        ]).T
    return prop


def stoich_crick_full():
    # species:
    # X0, X1, C00, C10, C11
    
    stoich = np.array([ 
                    [-1,+1,0,0,0,0],    # X0 -> X1
                    [+1,-1,0,0,0,0],    # X1 -> X0
                    [-2,0,+1,0,0,0],    # X0 + X0 -> C00
                    [-1,-1,0,+1,0,0],    # X0 + X1 -> C10
                    [0,-2,0,0,+1,0],    # X1 + X1 -> C11
                    [+2,0,-1,0,0,0],    # C00 -> X0 + X0
                    [+1,+1,0,-1,0,0],    # C10 -> X0 + X0
                    [0,+2,0,0,-1,0],    # C11 -> X1 + X1
                    [0,0,-1,+1,0,0],    # C00 -> C10 (2 eps)
                    [0,0,0,-1,+1,0],    # C10 -> C11 (eps + kcat)
                    [0,0,0,+1,-1,0],    # C11 -> C10 (2 delta)
                    [0,0,+1,-1,0,0],    # C10 -> C00 (delta)
                    [-1,0,0,0,0,+1],    # X0 -> Z
                    [0,-1,0,0,0,+1],    # X1 -> Z
                    [+1,0,0,0,0,-1],    # Z -> X0
                    ])
    return stoich

# ==================



def get_rxn_list_crick_full(params):
    alpha, beta = params['alpha'], params['beta']
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    bf, br = params['bf'], params['br']

    # Reactions as (nu, propensity)
    # species order: (X0, X1, C00, C10, C11, Z)
    reactions = []
    

    
    # 0) Z -> X0
    reactions.append(Reaction(np.array([+1,0,0,0,0,-1]),
                              lambda x: alpha * x[5]))
    
    # 1) X0 -> Z
    reactions.append(Reaction(np.array([-1,0,0,0,0,+1]),
                              lambda x: beta * x[0]))

    # 2) X1 -> Z
    reactions.append(Reaction(np.array([0,-1,0,0,0,+1]),
                              lambda x: beta * x[1]))
    
    # 3) X0 -> X1
    reactions.append(Reaction(np.array([-1,+1,0,0,0,0]),
                              lambda x: eps * x[0]))
    
    # 4) X1 -> X0
    reactions.append(Reaction(np.array([+1,-1,0,0,0,0]),
                              lambda x: gamma * x[1]))
    
    # 5) X0+X0 -> C00
    reactions.append(Reaction(np.array([-2,0,+1,0,0,0]),
                              lambda x: bf * x[0]*(x[0]-1)//2))
    
    # 6) X0+X1 -> C10
    reactions.append(Reaction(np.array([-1,-1,0,+1,0,0]),
                              lambda x: bf * x[0]*x[1]))
    
    # 7) X1+X1 -> C11
    reactions.append(Reaction(np.array([0,-2,0,0,+1,0]),
                              lambda x: bf * x[1]*(x[1]-1)//2))
    
    # 8) C00 -> X0+X0
    reactions.append(Reaction(np.array([+2,0,-1,0,0,0]),
                              lambda x: br * x[2]))
    
    # 9) C10 -> X1+X0
    reactions.append(Reaction(np.array([+1,+1,0,-1,0,0]),
                              lambda x: br * x[3]))
    
    # 10) C11 -> X1+X1
    reactions.append(Reaction(np.array([0,+2,0,0,-1,0]),
                              lambda x: br * x[4]))
    
    # 11) C00 -> C10 (2 eps)
    reactions.append(Reaction(np.array([0,0,-1,+1,0,0]),
                              lambda x: 2*eps * x[2]))
    
    # 12) C10 -> C11 (eps + kcat)
    reactions.append(Reaction(np.array([0,0,0,-1,+1,0]),
                              lambda x: (eps+k) * x[3]))
    
    # 13) C11 -> C10 (2 delta)
    reactions.append(Reaction(np.array([0,0,0,+1,-1,0]),
                              lambda x: 2*gamma * x[4]))
    
    # 14) C10 -> C00 (delta)
    reactions.append(Reaction(np.array([0,0,+1,-1,0,0]),
                              lambda x: gamma * x[3]))
    
    return reactions


def get_rxn_list_crick_binding(params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    bf, br = params['bf'], params['br']

    # Reactions as (nu, propensity)
    # species order: (X0, X1, C00, C10, C11)
    reactions = []
    

    
    # 3) X0 -> X1
    reactions.append(Reaction(np.array([-1,+1,0,0,0]),
                              lambda x: eps * x[0]))
    
    # 4) X1 -> X0
    reactions.append(Reaction(np.array([+1,-1,0,0,0]),
                              lambda x: gamma * x[1]))
    
    # 5) X0+X0 -> C00
    reactions.append(Reaction(np.array([-2,0,+1,0,0]),
                              lambda x: bf * x[0]*(x[0]-1)//2))
    
    # 6) X0+X1 -> C10
    reactions.append(Reaction(np.array([-1,-1,0,+1,0]),
                              lambda x: bf * x[0]*x[1]))
    
    # 7) X1+X1 -> C11
    reactions.append(Reaction(np.array([0,-2,0,0,+1]),
                              lambda x: bf * x[1]*(x[1]-1)//2))
    
    # 8) C00 -> X0+X0
    reactions.append(Reaction(np.array([+2,0,-1,0,0]),
                              lambda x: br * x[2]))
    
    # 9) C10 -> X1+X0
    reactions.append(Reaction(np.array([+1,+1,0,-1,0]),
                              lambda x: br * x[3]))
    
    # 10) C11 -> X1+X1
    reactions.append(Reaction(np.array([0,+2,0,0,-1]),
                              lambda x: br * x[4]))
    
    # 11) C00 -> C10 (2 eps)
    reactions.append(Reaction(np.array([0,0,-1,+1,0]),
                              lambda x: 2*eps * x[2]))
    
    # 12) C10 -> C11 (eps + kcat)
    reactions.append(Reaction(np.array([0,0,0,-1,+1]),
                              lambda x: (eps+k) * x[3]))
    
    # 13) C11 -> C10 (2 delta)
    reactions.append(Reaction(np.array([0,0,0,+1,-1]),
                              lambda x: 2*gamma * x[4]))
    
    # 14) C10 -> C00 (delta)
    reactions.append(Reaction(np.array([0,0,+1,-1,0]),
                              lambda x: gamma * x[3]))
    
    return reactions


def get_rxn_list_crick_variance(params):
    alpha, beta = params['alpha'], params['beta']
    eps, gamma, k = params['eps'], params['gamma'], params['k']


    # Reactions as (nu, propensity)
    # species order: (C00, C10, C11, Z)
    reactions = []
    

    
    # 0) Z -> C00
    reactions.append(Reaction(np.array([+1,0,0,-1]),
                              lambda x: alpha * x[3]))
    
    # 1) C00 -> Z
    reactions.append(Reaction(np.array([-1,0,0,+1]),
                              lambda x: beta * x[0]))

    # 2) C10 -> Z
    reactions.append(Reaction(np.array([0,-1,0,+1]),
                              lambda x: beta * x[1]))
    
    # 3) C11 -> Z
    reactions.append(Reaction(np.array([0,0,-1,+1]),
                              lambda x: beta * x[2]))
    
    # 11) C00 -> C10 (2 eps)
    reactions.append(Reaction(np.array([-1,+1,0,0]),
                              lambda x: 2*eps * x[0]))
    
    # 12) C10 -> C11 (eps + kcat)
    reactions.append(Reaction(np.array([0,-1,+1,0]),
                              lambda x: (eps+k) * x[1]))
    
    # 13) C11 -> C10 (2 delta)
    reactions.append(Reaction(np.array([0,+1,-1,0]),
                              lambda x: 2*gamma * x[2]))
    
    # 14) C10 -> C00 (delta)
    reactions.append(Reaction(np.array([+1,-1,0,0]),
                              lambda x: gamma * x[1]))
    
    return reactions




def get_rxn_list_crick(params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']


    # Reactions as (nu, propensity)
    # species order: (C00, C10, C11, Z)
    reactions = []
    

    # 11) C00 -> C10 (2 eps)
    reactions.append(Reaction(np.array([-1,+1,0]),
                              lambda x: 2*eps * x[0]))
    
    # 12) C10 -> C11 (eps + kcat)
    reactions.append(Reaction(np.array([0,-1,+1]),
                              lambda x: (eps+k) * x[1]))
    
    # 13) C11 -> C10 (2 delta)
    reactions.append(Reaction(np.array([0,+1,-1]),
                              lambda x: 2*gamma * x[2]))
    
    # 14) C10 -> C00 (delta)
    reactions.append(Reaction(np.array([+1,-1,0]),
                              lambda x: gamma * x[1]))
    
    return reactions




# ===========

# UNUSED REACTION LISTS; THEY WERE USED FOR TESTING/DEBUGGING



def get_rxn_list_birthdeath(params):
    alpha, gamma = params['alpha'], params['gamma']
    
    reactions = []
    
    # 1) ∅ -> X
    reactions.append(Reaction(np.array([+1,0]), lambda x: alpha))
    
    # 2) X -> ∅
    reactions.append(Reaction(np.array([-1,0]), lambda x: gamma*x[0]))
    return reactions


def get_rxn_list_twostate(params):
    eps, gamma = params['eps'], params['gamma']
    
    # Reactions as (nu, propensity)
    # species order: (X0, X1)
    reactions = []
    

    # 1) X0 -> X1
    reactions.append(Reaction(np.array([-1,+1]),
                              lambda x: eps * x[0]))
    
    # 2) X1 -> X0
    reactions.append(Reaction(np.array([+1,-1]),
                              lambda x: gamma * x[1]))
    return reactions




def get_rxn_list_twostate_bd(params):
    alpha, beta = params['alpha'], params['beta']
    eps, delta = params['eps'], params['delta']

    # Reactions as (nu, propensity)
    # species order: (X0, X1)
    reactions = []
    

    # 1) X0 -> X1
    reactions.append(Reaction(np.array([-1,+1]),
                              lambda x: eps * x[0]))
    
    # 2) X1 -> X0
    reactions.append(Reaction(np.array([+1,-1]),
                              lambda x: delta * x[1]))
    
    # 3) ∅ -> X0
    reactions.append(Reaction(np.array([+1,0]),
                              lambda x: alpha))

    # 4) X0 -> ∅
    reactions.append(Reaction(np.array([-1,0]),
                              lambda x: beta * x[0]))
    
    # 5) X1 -> ∅
    reactions.append(Reaction(np.array([0,-1]),
                              lambda x: beta * x[1]))
    return reactions



