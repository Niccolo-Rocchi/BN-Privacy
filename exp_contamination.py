# Libraries
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import pyagrum as gum
from pathlib import Path
from pandarallel import pandarallel
import numpy as np
from tqdm import tqdm
import warnings
from utils import *


if __name__ == "__main__":

    print("--- Start ---")
    warnings.filterwarnings('ignore')

    # Init hyperparameters
    # print("Initialize hyperparameters ...", end=" ")
    n_nodes = 50                    # Number of nodes
    n_arcs  = 70                    # Number of edges
    n_modmax = 2                    # Max number of modalities per node
    gpop_ss = 10000                 # Sample size (general population)
    ratio = 10                      # Sample sizes ratio (i.e., pool : reference population = 1 : ratio)
    eps = 0.01                      # Contamination for CN
    n_bns = 5                       # Number of vertices BNs to extract from CN simplex
    error = np.arange(0, 1, 0.05)   # Error (alpha) range
    # print("[OK]")

    # Init parallelization
    # print("Initialize parallelization on ALL cores ...", end=" ")
    pandarallel.initialize(progress_bar=False, verbose=1)   # Show only warnings
    # print("[OK]")

    # Generate ground-truth BN
    # print("Generate BN ...", end=" ")
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(n_nodes=n_nodes, n_arcs=n_arcs, n_modmax=n_modmax)
    # print("[OK]")

    # Sample data   
    # print("Sample data ...", end=" ")
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()
    gpop = data_gen.to_pandas()

    pool_ss = gpop_ss // (ratio + 1)
    pool_idx = np.random.choice(gpop_ss, replace=False, size=pool_ss)
    pool = gpop.iloc[pool_idx]
    rpop = gpop.iloc[~ gpop.index.isin(pool_idx)]
    # print("[OK]")

    # Debug
    # assert(gpop_ss == gpop.shape[0])
    # assert(pool.shape[0] + rpop.shape[0] == gpop_ss)

    # Estimate BN(theta) from rpop and BN(theta_hat) from pool
    print("Learn BNs from pool and reference population ...", end=" ")
    theta_learner=gum.BNLearner(rpop)
    theta_learner.useSmoothingPrior(1e-5)
    bn_theta = theta_learner.learnParameters(bn.dag())

    theta_hat_learner=gum.BNLearner(pool)
    theta_hat_learner.useSmoothingPrior(1e-5)
    bn_theta_hat = theta_hat_learner.learnParameters(bn.dag())
    print("[OK]")

    ## Estimate CN by contamination
    # print(f"Estimate CN by contamination with eps={eps} ...", end=" ")

    # Init min & max BNs
    bn_min=gum.BayesNet(bn)
    bn_max=gum.BayesNet(bn)
    for n in bn.nodes():
        x=eps*min(bn.cpt(n).min(),1-bn.cpt(n).max())
        bn_min.cpt(n).translate(-x)
        bn_max.cpt(n).translate(x)

    # Create CN
    cn=gum.CredalNet(bn_min,bn_max)
    cn.intervalToCredal()
    # print("[OK]")

    # Extract random subset within simplex
    print(f"Extract {n_bns} BNs from credal set ...", end=" ")
    bns_sample = get_simplex_inner(cn, n_bns)
    print("[OK]", end=" ")
    are_all_bn_different(bns_sample)

    ## MIA (theoretical)
    # print(f"Compute theoretical power ...", end=" ")
    # Set ground truth membership
    gpop["in-pool"] = False
    gpop.loc[pool_idx, "in-pool"] = True
    tp = len(pool_idx)

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]
    # print("[OK]")

    # Init results
    results = pd.DataFrame(
        {"error": error,
        "power_bound": beta}
    )

    ## MIA (BN)
    print(f"Compute BN power ...", end=" ")
    # Estimate the distribution of LLR(x) from rpop (i.e. under H_0)
    bn_theta_ie = gum.LazyPropagation(bn_theta)
    bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
    L_im = rpop.parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1).dropna().sort_values() 

    # Compute LLR(x) on general population
    llr_gpop = gpop[[*bn.names()]].parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1)

    # Init the power vector (BN)
    power_bn = []

    # For each error ...
    for e in error:

        # Compute threshold
        t = np.quantile(L_im, e).item()

        # LLR test. Reject H_0? True => x in pool; False => x in rpop.
        y_pred = llr_gpop < t

        # Compute and store power (tpr)
        tpr = sum(gpop["in-pool"] & y_pred) / tp
        power_bn = power_bn + [tpr]
    print("[OK]")

    # Store results
    results["power_BN"] = power_bn

    ## MIA (CN)
    print(f"Compute CN power on {n_bns} BN vertices...")
    for i, bn_vertex in enumerate(tqdm(bns_sample, unit="item", dynamic_ncols=True)):

        # Estimate the distribution of LLR(x) from rpop (i..e under H_0)
        bn_vertex_ie = gum.LazyPropagation(bn_vertex)
        L_im_vertex = rpop.parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_vertex_ie), axis=1).dropna().sort_values()

        # Compute LLR(x) on general population
        llr_gpop_vertex = gpop[[*bn.names()]].parallel_apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_vertex_ie), axis=1)

        # Init the power vector (BN vertex)
        power_bn_vertex = []

        # For each error ...
        for e in error:

            # Compute threshold
            t = np.quantile(L_im_vertex, e).item()

            # LLR test. Reject H_0? True => x in pool; False => x in rpop.
            y_pred_vertex = llr_gpop_vertex < t

            # Compute and store power (tpr)
            tpr = sum(gpop["in-pool"] & y_pred_vertex) / tp
            power_bn_vertex = power_bn_vertex + [tpr]
        
        # Store results
        results[f"power_BN_v_{i}"] = power_bn_vertex

    # Save results
    # print("Save results ...", end=" ")
    results_path = Path("./results")
    results_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(f"./results/compl{compl}-eps{eps}.csv")
    # print("[OK]")
    print("--- Quit ---")