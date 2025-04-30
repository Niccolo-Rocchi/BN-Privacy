import math
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy.stats import norm
import pyagrum as gum

from utils import *

warnings.filterwarnings('ignore')

# Run local IDM experiment
def run_idm(conf):

    # Init global hyperp.
    gpop_ss = 10000
    rpop_ss = 5000
    pool_ss = 500
    n_ds = 30
    n_bns = 500
    error = np.logspace(-4, 0, 30, endpoint=False)
    
    # Init local hyperp.
    exp = conf[0]
    gpop_or = pd.read_csv(f"./data/{exp}.csv")
    bn_or = gum.loadBN(f"./bns/{exp}.bif")
    n_nodes = len(bn_or.nodes())
    ess = conf[1].get("ess")

    # Debug
    assert(gpop_ss == gpop_or.shape[0])
    assert(n_nodes == gpop_or.shape[1])

    # Compute theoretical bound
    compl = bn_or.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Init results
    results = pd.DataFrame(
        {"error": error,
        "power_bound": beta}
    )

    for ds in range(n_ds):
        
        try:

            # Copy BN and data
            gpop = gpop_or.copy()
            bn = gum.BayesNet(bn_or)

            # Sample pool and rpop
            pool_idx = np.random.choice(range(gpop_ss), size=pool_ss, replace=False)
            pool = gpop.iloc[pool_idx]
            rpop_idx = np.random.choice(range(gpop_ss), size=rpop_ss, replace=False)
            rpop = gpop.iloc[rpop_idx]

            # Set ground truth membership
            gpop["in-pool"] = False
            gpop.loc[pool_idx, "in-pool"] = True
            tp = len(pool_idx)

            # Estimate BN(theta) from rpop and BN(theta_hat) from pool
            theta_learner=gum.BNLearner(rpop)
            theta_learner.useSmoothingPrior(1e-5)
            bn_theta = theta_learner.learnParameters(bn.dag())

            theta_hat_learner=gum.BNLearner(pool)
            theta_hat_learner.useSmoothingPrior(1e-5)
            bn_theta_hat = theta_hat_learner.learnParameters(bn.dag())

            # Estimate CN by local IDM 
            bn_copy = gum.BayesNet(bn)
            add_counts(bn_copy, pool)
            cn = gum.CredalNet(bn_copy)
            cn.idmLearning(ess)

            # Extract random subset within simplex
            bns_sample = get_simplex_inner(cn, n_bns)
            # are_all_bn_different(bns_sample)

            # MIA (BN)
            # Estimate the distribution of LLR(x) from rpop (i.e. under H_0)
            bn_theta_ie = gum.LazyPropagation(bn_theta)
            bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
            L_im = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1).dropna().sort_values() 

            # Compute LLR(x) on general population
            llr_gpop = gpop[[*bn.names()]].apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_theta_hat_ie), axis=1)

            # Init the power vector (BN)
            power_bn = []

            # For each error ...
            for e in error:

                # Compute threshold
                t = np.quantile(L_im, 1-e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred = llr_gpop > t

                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred) / tp
                power_bn = power_bn + [tpr]

            # Store BN results for this data sample
            results[f"power_BN_ds{ds}"] = power_bn

            # MIA (CN)
            best_sum = 0
            best_bn_idx = 0
            for i, bn_s in enumerate(bns_sample):

                # Estimate the likelihood of rpop
                bn_s_ie = gum.LazyPropagation(bn_s)
                L_im_s = rpop.apply(lambda x: LL(x.to_dict(), bn_s_ie), axis=1).dropna()

                L_im_s_sum = np.sum(L_im_s)
                if L_im_s_sum > best_sum: 
                    best_sum = L_im_s_sum
                    best_bn_idx = i

            # Get best BN inside simplex
            best_bn_s_ie = gum.LazyPropagation(bns_sample[best_bn_idx])
            L_im_best = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, best_bn_s_ie), axis=1).dropna()

            # Compute LLR(x) on general population
            llr_gpop_s = gpop[[*bn.names()]].apply(lambda x: LLR(x.to_dict(), bn_theta_ie, best_bn_s_ie), axis=1)

            # Init the power vector (CN)
            power_cn = []

            # For each error ...
            for e in error:

                # Compute threshold
                t = np.quantile(L_im_best, 1-e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred_s = llr_gpop_s > t
                
                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred_s) / tp
                power_cn = power_cn + [tpr]
            
            # Store CN results for this data sample
            results[f"power_CN_ds{ds}"] = power_cn

        except:

            with open("./results/log.txt", "a") as log: 
                log.write(f"{exp}: error with sample {ds}.\n")
                # log.write(traceback.format_exc())

    # Save results
    results.to_csv(f"./results/{exp}-ess{ess}.csv", index=False)
