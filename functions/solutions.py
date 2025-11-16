import numpy as np
from scipy.stats import binom
from scipy.stats import multinomial




# utility function: outputs all possible tuples of nonnegative integers with x0 + x1 + x2 = N
def triples_sum_to(n):
    x_ = []
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            x_.append([a, b, c])
    return np.array(x_)


# ==============

# MODEL SOLUTIONS


# Pure turnover model:  steady state (p_z_given_0) and time-dependent probabilities (p_z_given_1) 
def get_probs_noisefree_twostate(t, params):
    gamma = params['gamma']
    
    # steady state probability distribution
    pss_0 = 1
    pss_1 = 0
    p_z_given_0 = np.array([ pss_0, pss_1 ])
    
    # transition probability
    p_0 = 1 - np.exp(- gamma*t)
    p_1 = np.exp(- gamma*t)
    p_z_given_1 = np.array([ p_0, p_1 ]).T
    
    return p_z_given_0, p_z_given_1


# Two-state model:  steady state (p_z_given_0) and time-dependent probabilities (p_z_given_1) 
def get_probs_twostate(t, params):
    eps, gamma = params['eps'], params['gamma']
    
    lamb = eps + gamma
    
    # steady state probability distribution
    pss_0 = gamma/lamb
    pss_1 = eps/lamb
    p_z_given_0 = np.array([ pss_0, pss_1 ])
    
    # transition probability
    p_0 = (gamma/lamb)*( 1 - np.exp(- lamb*t) )
    p_1 = (eps/lamb) + (gamma/lamb)*np.exp(- lamb*t)
    p_z_given_1 = np.array([ p_0, p_1 ]).T
    
    return p_z_given_0, p_z_given_1


# Pair model:  steady state (p_z_given_0) and time-dependent probabilities (p_z_given_1) 
def get_probs_twotwostate(t, params):
    eps, gamma = params['eps'], params['gamma']
    
    lamb = eps + gamma
    
    
    # steady state probability distribution
    pss_0 = gamma/lamb
    pss_1 = eps/lamb
    
    pss_minus = pss_0**2
    pss_mid = 2*pss_0*pss_1
    pss_plus = pss_1**2
    p_z_given_0 = np.array([ pss_minus, pss_mid, pss_plus])
    
    
    # transition probability
    p_0 = (gamma/lamb)*( 1 - np.exp(- lamb*t) )
    p_1 = (eps/lamb) + (gamma/lamb)*np.exp(- lamb*t)
    
    p_minus = p_0**2
    p_mid = 2*p_0*p_1
    p_plus = p_1**2
    p_z_given_1 = np.array([ p_minus, p_mid, p_plus] ).T
    
    return p_z_given_0, p_z_given_1


# N two-state models:  steady state (p_z_given_0) and time-dependent probabilities (p_z_given_1) 
def get_probs_ntwostate(t, params):
    eps, gamma, N = params['eps'], params['gamma'], params['N']
    
    lamb = eps + gamma
    
    # 'heads' probabilities
    p_inf = eps/lamb
    p_t = (eps/lamb) + (gamma/lamb)*np.exp(- lamb*t)
    
    # get steady state and transition probability via binomial distribution call
    x = np.arange(0, N+1)
    p_z_given_0 = binom.pmf( x, N, p_inf)
    p_z_given_1 = binom.pmf( x[:,None], N, p_t[None,:]).T
    
    return p_z_given_0, p_z_given_1


# Crick switch:  steady state (p_z_given_0) and time-dependent probabilities (p_z_given_1) 
def get_probs_crick(t, params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']

    # important parameter combinations
    Delta = 0.5*(  np.sqrt( (eps + gamma + k)**2 + 4*k*(gamma - eps) ) - (eps + gamma + k)  )
    lamb_1 = eps + gamma - Delta
    lamb_2 = 2*(eps + gamma) + Delta + k
    
    # steady state probability distribution
    pss_norm = (eps + gamma)**2 + k*eps
    pss_minus = gamma**2
    pss_mid = 2*gamma*eps
    pss_plus = eps*(eps + k)
    p_z_given_0 = np.array([ pss_minus, pss_mid, pss_plus])/pss_norm
    
    
    # transition probability
    c_norm = pss_norm*( eps + gamma + 2*Delta + k )
    c1 = gamma*( 2*eps*(eps + gamma) + (gamma + 2*eps)*Delta + k*gamma  )/c_norm
    c2 = gamma*( gamma*(eps + gamma) + (gamma + 2*eps)*Delta + 2*k*eps )/c_norm
    
    v1 = np.array([ -(gamma + Delta + k)/(eps + k) , (gamma - eps + Delta)/(eps + k) , 1 ])
    v2 = np.array([ (eps + Delta)/(eps + k), -(1 + (eps + Delta)/(eps + k)), 1 ])
     
    p_z_given_1 = p_z_given_0[None,:] + c1*v1[None,:]*np.exp(-lamb_1*t[:,None]) + c2*v2[None,:]*np.exp(-lamb_2*t[:,None])
    
    return p_z_given_0, p_z_given_1


# N Crick switches:  steady state (p_z_given_0) and time-dependent probabilities (p_z_given_1) 
def get_probs_ncrick(t, params):
    N = params['N']
    
    pi, p_t = get_probs_crick(t, params)


    x_ = triples_sum_to(N)    # get all possible states, i.e., tuples of nonnegative integers with x0 + x1 + x2 = N


    logpmf = multinomial.logpmf(x_[:, None, :], n=N, p=p_t[None, :, :])  # (num_states, num_params)
    pmf   = np.exp(logpmf).T     


    logpmf_ss = multinomial.logpmf(x_[:, None, :], n=N, p=pi[None, :])
    pmf_ss = np.exp(logpmf_ss)[:,0]
    return pmf_ss, pmf


# N Crick switches, same as above but not including the tripes_sum_to call to save a small amount of compute
def get_probs_ncrick_shortcut(t, params, x_):
    N = params['N']
    
    pi, p_t = get_probs_crick(t, params)


    logpmf = multinomial.logpmf(x_[:, None, :], n=N, p=p_t[None, :, :])  # (num_states, num_params)
    pmf   = np.exp(logpmf).T     


    logpmf_ss = multinomial.logpmf(x_[:, None, :], n=N, p=pi[None, :])
    pmf_ss = np.exp(logpmf_ss)[:,0]
    return pmf_ss, pmf


# ==========================================

# EIGENVECTORS AND EIGENVALUES



# Pair model: eigenvectors
def get_eigenvectors_twotwostate(params):
    eps, gamma = params['eps'], params['gamma']
    

    # steady state probability distribution
    pss_norm = (eps + gamma)**2
    pss_minus = gamma**2
    pss_mid = 2*gamma*eps
    pss_plus = eps**2
    v0 = np.array([ pss_minus, pss_mid, pss_plus])/pss_norm
    
    # slow eigenvector
    v1 = np.array([gamma, - gamma + eps, - eps])/gamma
    
    # fast eigenvector
    v2 = np.array([ 0.5, -1, 0.5  ])
    
    eigenvectors = {'v0':v0, 'v1':v1, 'v2':v2}
    return eigenvectors


# Crick switch: eigenvectors
def get_eigenvectors_crick(params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    
    # important parameter combinations
    Delta = 0.5*(  np.sqrt( (eps + gamma + k)**2 + 4*k*(gamma - eps) ) - (eps + gamma + k)  )
    
    # steady state probability distribution
    pss_norm = (eps + gamma)**2 + k*eps
    pss_minus = gamma**2
    pss_mid = 2*gamma*eps
    pss_plus = eps*(eps + k)
    v0 = np.array([ pss_minus, pss_mid, pss_plus])/pss_norm
    
    # slow eigenvector
    v1 = np.array([gamma, - gamma + eps + Delta, - eps - Delta])/gamma
    
    # fast eigenvector
    v2 = np.array([ 0.5*(1 - Delta/(gamma - eps)), -1, 0.5*(  1 + Delta/(gamma - eps)  )  ])
    
    eigenvectors = {'v0':v0, 'v1':v1, 'v2':v2}
    return eigenvectors


# Crick switch: eigenvalues
def get_eigenvalues_crick(params):
    eps, gamma, k = params['eps'], params['gamma'], params['k']
    
    # important parameter combinations
    Delta = 0.5*(  np.sqrt( (eps + gamma + k)**2 + 4*k*(gamma - eps) ) - (eps + gamma + k)  )

    lamb_1 = eps + gamma - Delta
    lamb_2 = 2*(eps + gamma) + Delta + k
    
    tau_1 = 1/lamb_1
    tau_2 = 1/lamb_2
    
    eigenvalues = {'lamb_1':lamb_1, 'lamb_2':lamb_2, 'tau_1':tau_1, 'tau_2':tau_2}
    return eigenvalues


# ================


# Mean and variance of N two-state models
def get_mean_var_ntwostate(t, params):
    eps, gamma, N = params['eps'], params['gamma'], params['N']
    
    lamb = eps + gamma
    
    x = (eps/lamb) + (gamma/lamb)*np.exp(- lamb*t)
    sigma = np.sqrt(x*(1-x)/N)
    return x, sigma




