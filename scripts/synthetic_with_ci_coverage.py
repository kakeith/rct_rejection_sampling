"""
Synthetic experiments

With CI coverage results
"""

from causal_eval.sampling import *
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

if __name__ == "__main__":
    # NUM_CORES = -1 #use all the cores on big cluster 
    # NUM_BOOTSTRAP_SAMPLES = 1000
    # NUM_RANDOM_SEEDS = 1000 #random seeds for DGP 
    # NUM_SAMPLES =100000 # in DGP

    #Settings just for code development checks  
    NUM_CORES = 10 #use all the cores on big cluster 
    NUM_BOOTSTRAP_SAMPLES = 1000
    NUM_RANDOM_SEEDS = 10 #random seeds for DGP 
    NUM_SAMPLES =1000 # in DGP

    # random seed for the sampling methods same throughout
    rng = np.random.default_rng(10)

    ### SYNTHETIC SETTING 1 ==> just the plots 
    confound_func_params={"para_form": "linear"} 
    for num_samples in [100000, 3000]: 
        
        data, rct_ace = synthetic_dgp(setting=1, num_samples=num_samples)

        data_out1 = bootstrapping_three_methods_linear(data, rct_ace, confound_func_params)
        title = f"Synthetic DGP #1 \nSize of RCT data = {len(data)} \n1000 Bootstrap samples"
        fig, ax = plot_bootstrap(data_out1, rct_ace, title)
        fig.savefig(f"syn1-ci-{num_samples}.png")

    ### SYNTHETIC SETTING 1
    setting=1
    confound_func_params={"para_form": "linear"} 
    print(f"==== SYNTHETIC SETTING {setting} ===")  
    data, rct_ace = synthetic_dgp(setting=setting, num_samples=NUM_SAMPLES)
    results =  many_seeds(data, rct_ace, confound_func_params, 
                        is_linear=True, 
                        num_seeds=NUM_RANDOM_SEEDS, 
                        has_bootstrap=True, 
                        num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES, 
                        num_cores=NUM_CORES,
                        run_in_parallel=True)

    #save file 
    fname= f"syn{setting}-results.pkl"
    print(f"Results (synthetic {setting}) saved to: ", fname)
    with open(fname, "wb") as file:
        pickle.dump(results, file)

    ### SYNTHETIC SETTING 2
    setting=2
    confound_func_params={"para_form": "linear"} 
    print(f"==== SYNTHETIC SETTING {setting} ===")  
    data, rct_ace = synthetic_dgp(setting=setting, num_samples=NUM_SAMPLES)
    results =  many_seeds(data, rct_ace, confound_func_params, 
                        is_linear=True, 
                        num_seeds=NUM_RANDOM_SEEDS, 
                        has_bootstrap=True, 
                        num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES, 
                        num_cores=NUM_CORES,
                        run_in_parallel=True)

    #save file 
    fname= f"syn{setting}-results.pkl"
    print(f"Results (synthetic {setting}) saved to: ", fname)
    with open(fname, "wb") as file:
        pickle.dump(results, file)

    ### SYNTHETIC SETTING 3
    setting=3
    confound_func_params = {"para_form": "nonlinear", "C1": 1.5, "C2": -0.7, "C3": 1.2, "C4": 1.5, "C5": -1.2}
    print(f"==== SYNTHETIC SETTING {setting} ===")  
    data, rct_ace = synthetic_dgp(setting=setting, num_samples=NUM_SAMPLES)
    results =  many_seeds(data, rct_ace, confound_func_params, 
                        is_linear=False, #note, different for synthetic setting 3  
                        num_seeds=NUM_RANDOM_SEEDS, 
                        has_bootstrap=True, 
                        num_bootstrap_samples=NUM_BOOTSTRAP_SAMPLES, 
                        num_cores=NUM_CORES,
                        run_in_parallel=True)

    #save file 
    fname= f"syn{setting}-results.pkl"
    print(f"Results (synthetic {setting}) saved to: ", fname)
    with open(fname, "wb") as file:
        pickle.dump(results, file)




