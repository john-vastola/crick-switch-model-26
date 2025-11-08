import numpy as np



# vectorized version of np.random.choice
# assumes input of shape (num_runs, num_possibilities)
# for each run, chooses possibility by interpreting the second axis as describing prob weights
# output is shape (num_runs)
def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)



# D molecule types and M possible rxns
# x0: vector of initial conditions, (num_samples, D)
# prop: function that take a state (num_samples, D) as input and output a propensity (num_samples, M)
# stoich: stoichiometry matrix, (M, D)
# t_sim: total simulation time
def simulate_ssa(x0, prop, stoich, num_steps):
    
    num_samples, D = x0.shape 
    t_rec = np.zeros((num_steps+1, num_samples))    # init recording times   
    counts = np.zeros((num_steps + 1, num_samples, D)); counts[0] = x0    # init counts matrix
       
    state = np.copy(x0); current_time = np.zeros(num_samples)   # initialize state and current time; (num_samples, D)
    for i in range(num_steps):   # while at least one time is less than max sim time
        
        rates = prop(state)    # compute rates given current state, output shape (num_samples, M)
        next_rxn = random_choice_prob_index(rates/(np.sum(rates, axis=1)[:,None]))    # randomly draw transition given normed rates; (num_samples)
        dt = np.random.exponential(scale=1/np.sum(rates,axis=1), size=num_samples)   # randomly draw next rxn times; (num_samples)
        
        current_time += dt; state += stoich[next_rxn,:] # modify times, states
        counts[i+1] = state; t_rec[i+1] = current_time

    return t_rec, counts   

