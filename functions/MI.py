import numpy as np





# decoder-independent mutual info between p_z_given_0 (steady state) and p_z_given_1 (time-dependent distribution)
def get_MI(p_z_given_0, p_z_given_1, prior_0 = 0.5):
    """
    :p_z_given_0: steady state prob of z given u = 0 at all times ... shape = (num_states)
    :p_z_given_1: prob of z given nontrivial stimulus, shape = (num_t, num_states)
    """
    prior_1 = 1 - prior_0
    p_z_given_0 = p_z_given_0[None,:]   # expand dims
    

    small = 1e-12   # too much smaller, start getting errors
    MI = (  prior_0*p_z_given_0*np.log2( p_z_given_0/(prior_0*p_z_given_0 + prior_1*p_z_given_1 + small) + small ) 
          + prior_1*p_z_given_1*np.log2( p_z_given_1/(prior_0*p_z_given_0 + prior_1*p_z_given_1 + small) + small ) )
    MI = np.sum(MI , axis=-1)
    return MI






def get_capacity(p):
    return ( 1 - 0.5*(  - p*np.log2(p) + (1 + p)*np.log2(1 + p)  )  )


def get_t_life(t, params_list, get_probs, get_MI):
    num_things = len(params_list)
    t_life = np.zeros(num_things)

    for i in range(num_things):
        p_z_given_0, p_z_given_1 = get_probs(t, params_list[i])
        MI = get_MI(p_z_given_0, p_z_given_1)
 
        t_life[i] = t[np.argwhere(MI <= 0.5*MI[0])[0][0]]
    return t_life


def get_performances(t, params_list, get_probs, get_MI, tau=1):
    num_things = len(params_list)
    J = np.zeros(num_things)

    for i in range(num_things):
        p_z_given_0, p_z_given_1 = get_probs(t, params_list[i])
        MI = get_MI(p_z_given_0, p_z_given_1)
        J[i] = np.trapz(MI*np.exp(-t/tau), x=t  )
    return J/tau




def get_MI_2D(t, k, N, eps, gamma, x_, get_probs_ncrick_shortcut, get_MI):
    num_k, num_N = len(k), len(N)
    MI = np.zeros((num_k, num_N, len(t)))

    for i in range(0, num_k):
        for j in range(0, num_N):
            print('k: '+str(k[i])+', N: '+str(N[j]))
            params = {'eps':eps, 'gamma':gamma, 'k':k[i], 'N':N[j]}
            p_z_given_0, p_z_given_1 = get_probs_ncrick_shortcut(t, params, x_[j])
            MI[i,j,:] = get_MI(p_z_given_0, p_z_given_1)
    return MI


def get_t_life_2D(t, k, N, MI):
    num_k, num_N = len(k), len(N)
    t_life = np.zeros((num_k, num_N))

    for i in range(0, num_k):
        for j in range(0, num_N):
            t_life[i,j] = t[np.argwhere(MI[i,j] <= 0.5*MI[i,j][0])[0][0]]
    return t_life


def get_performances_2D(t, k, N, MI, tau=1):
    num_k, num_N = len(k), len(N)
    J = np.zeros((num_k, num_N))

    for i in range(0, num_k):
        for j in range(0, num_N):
            J[i,j] = np.trapz(MI[i,j,:]*np.exp(-t/tau), x=t  )
    return J/tau




