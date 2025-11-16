import numpy as np




# Crick switch: average first-passage time
def get_FPavg_crick(k, params):
    eps, gamma = params['eps'], params['gamma']

    
    t_avg = (3*gamma + eps + k)/(2*gamma*gamma)
    return t_avg




# ===========

# UNUSED




# Two-state model: 
def get_FPdist_twostate(t, params):
    """
    Returns first-passage time distribution of two-state model.

    Parameters
    ----------
    t : ndarray of float, shape (num_t,)
        Array of times.

    Returns
    -------
    f_t : ndarray of float, shape (num_t,)
          First-passage time distribution f(t), evaluated at time points specified by t.
    """
    
    gamma = params['gamma']
    
    f_t = gamma*np.exp(-gamma*t)
    return f_t


def get_FPdist_twotwostate(t, params):
    """
    Returns first-passage time distribution of pair of two-state models.

    Parameters
    ----------
    t : ndarray of float, shape (num_t,)
        Array of times.

    Returns
    -------
    f_t : ndarray of float, shape (num_t,)
          First-passage time distribution f(t), evaluated at time points specified by t.
    """
    
    eps, gamma = params['eps'], params['gamma']
    
    # relevant parameter combinations
    Delta = 0.5*( np.sqrt( (eps + gamma)**2 + 4*eps*gamma ) - (eps + gamma) )
    lamb_1 = gamma - Delta
    lamb_2 = 2*gamma + Delta + eps
    lamb_diff = gamma + 2*Delta + eps
    
    # FP dist is normalized difference of exponentials
    f_t = ((2*gamma*gamma)/lamb_diff)*( np.exp(-lamb_1*t) - np.exp(-lamb_2*t) )
    return f_t


def get_FPdist_crick(t, params):
    """
    Returns first-passage time distribution of Crick model.

    Parameters
    ----------
    t : ndarray of float, shape (num_t,)
        Array of times.

    Returns
    -------
    f_t : ndarray of float, shape (num_t,)
          First-passage time distribution f(t), evaluated at time points specified by t.
    """
    
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    
    # relevant parameter combinations
    Delta = 0.5*( np.sqrt( (eps + k + gamma)**2 + 4*(eps + k)*gamma ) - (eps + k + gamma) )
    lamb_1 = gamma - Delta
    lamb_2 = 2*gamma + Delta + eps + k
    lamb_diff = gamma + 2*Delta + eps + k
    
    # FP dist is normalized difference of exponentials
    f_t = ((2*gamma*gamma)/lamb_diff)*( np.exp(-lamb_1*t) - np.exp(-lamb_2*t) )
    return f_t



