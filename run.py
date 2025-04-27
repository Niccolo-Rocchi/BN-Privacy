import math
from pathlib import Path
import warnings
import traceback

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from scipy.stats import norm
from tqdm import tqdm
import pyagrum as gum
import yaml

from utils import *

warnings.filterwarnings('ignore')

def run_exp_idm(inv_configs, conf):

    # Init hyperparameters
    n_ds = inv_configs["n_ds"]
    n_modmax = inv_configs["n_modmax"]
    gpop_ss = inv_configs["gpop_ss"]
    rpop_ss = inv_configs["rpop_ss"]
    pool_ss = inv_configs["pool_ss"]
    error = eval(inv_configs["error"])
    n_bns = inv_configs["n_bns"]
    n_nodes = conf["n_nodes"]
    edge_ratio = conf["edge_ratio"]
    ess = conf["ess"]

    # Generate ground-truth BN
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(n_nodes=n_nodes, n_arcs=int(n_nodes * edge_ratio), n_modmax=n_modmax)

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Init results
    results = pd.DataFrame(
        {"error": error,
        "power_bound": beta,
        "power_BN": np.zeros(len(error)),
        "power_CN": np.zeros(len(error))}
    )

    # Sample gpop
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()
    gpop = data_gen.to_pandas()

    for ds in tqdm(range(n_ds), unit="item", desc="Data samples", dynamic_ncols=True):
        
        try:

            # Init results of specific data sample
            results_ds = pd.DataFrame({"error": error})

            # Sample pool and rpop
            gpop = gpop[[*bn.names()]]
            pool_idx = np.random.choice(range(gpop_ss), size=pool_ss, replace=False)
            pool = gpop.iloc[pool_idx]
            rpop_idx = np.random.choice(range(gpop_ss), size=rpop_ss, replace=False)
            rpop = gpop.iloc[rpop_idx]

            # Debug
            # assert(gpop.shape[0] == gpop_ss)
            # assert(pool.shape[0] == pool_ss)
            # assert(rpop.shape[0] == rpop_ss)

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

            # Debug
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
                t = np.quantile(L_im, e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred = llr_gpop < t

                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred) / tp
                power_bn = power_bn + [tpr]

            # Store BN results for this data sample
            results_ds["power_BN"] = power_bn

            # MIA (CN)
            best_sum = 0
            best_bn_idx = 0
            for i, bn_s in enumerate(tqdm(bns_sample, desc="Credal samples", unit="item", dynamic_ncols=True, leave=False)):

                # Estimate the distribution of LLR(x) from rpop (i..e under H_0)
                bn_s_ie = gum.LazyPropagation(bn_s)
                L_im_s = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_s_ie), axis=1).dropna()

                L_im_s_sum = np.sum(L_im_s)
                if L_im_s_sum > best_sum: 
                    best_sum = L_im_s_sum
                    best_bn_idx = i

            # Get best BN inside simplex
            best_bn_s_ie = gum.LazyPropagation(bns_sample[best_bn_idx])
            L_im_best = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, best_bn_s_ie), axis=1).dropna()

            # Compute LLR(x) on general population
            llr_gpop_s = gpop[[*bn.names()]].apply(lambda x: LLR(x.to_dict(), bn_theta_ie, best_bn_s_ie), axis=1)

            # Init the power vector (BN vertex)
            power_bn_s = []

            # For each error ...
            for e in error:

                # Compute threshold
                t = np.quantile(L_im_best, e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred_s = llr_gpop_s < t
                
                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred_s) / tp
                power_bn_s = power_bn_s + [tpr]
            
            # Store CN results for this data sample
            results_ds["power_CN"] = power_bn_s

            # Add ds results to final results
            results["power_BN"] += results_ds["power_BN"]
            results["power_CN"] += results_ds["power_CN"]
            # results_ds.drop(["error", "power_BN"], axis=1, inplace=True)

            # results["CN_avg"] += results_ds.mean(axis=1)
            # results["CN_min"] += results_ds.min(axis=1)
            # results["CN_max"] += results_ds.max(axis=1)

            # Debug
            # assert(results_ds.shape[1] == n_bns)
            # assert(results.shape[1] == 6)
        except:
            with open("./results/idm/log.txt", "a") as log: log.write(f"{conf['meta']}: error with sample {ds}.\n")

                # Debug
                # log.write(traceback.format_exc())

    # Compute average results and save them
    results[["power_BN", "power_CN"]] /= n_ds
    results.to_csv(f"./results/idm/{conf['meta']}-compl{compl}.csv", index=False)
    with open("./results/idm/log.txt", "a") as log: log.write(f"{conf['meta']}: [OK]\n")


def run_exp_cont(inv_configs, conf):
    
    # Init hyperparameters
    n_ds = inv_configs["n_ds"]
    n_modmax = inv_configs["n_modmax"]
    gpop_ss = inv_configs["gpop_ss"]
    rpop_ss = inv_configs["rpop_ss"]
    pool_ss = inv_configs["pool_ss"]
    error = eval(inv_configs["error"])
    n_bns = inv_configs["n_bns"]
    n_nodes = conf["n_nodes"]
    edge_ratio = conf["edge_ratio"]
    eps = conf["eps"]

    # Generate ground-truth BN
    bn_gen = gum.BNGenerator()
    bn = bn_gen.generate(n_nodes=n_nodes, n_arcs=int(n_nodes * edge_ratio), n_modmax=n_modmax)

    # Compute theoretical bound
    compl = bn.dim()
    bound = math.sqrt(compl/pool_ss)

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in error]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Init results
    results = pd.DataFrame(
        {"error": error,
        "power_bound": beta,
        "power_BN": np.zeros(len(error)),
        "power_CN": np.zeros(len(error))}
    )

    # Sample gpop
    data_gen = gum.BNDatabaseGenerator(bn)
    data_gen.drawSamples(gpop_ss)
    data_gen.setDiscretizedLabelModeRandom()
    gpop = data_gen.to_pandas()

    for ds in tqdm(range(n_ds), unit="item", desc="Data samples", dynamic_ncols=True):
        
        try:

            # Init results of specific data sample
            results_ds = pd.DataFrame({"error": error})

            # Sample pool and rpop
            gpop = gpop[[*bn.names()]]
            pool_idx = np.random.choice(range(gpop_ss), size=pool_ss, replace=False)
            pool = gpop.iloc[pool_idx]
            rpop_idx = np.random.choice(range(gpop_ss), size=rpop_ss, replace=False)
            rpop = gpop.iloc[rpop_idx]

            # Debug
            # assert(gpop.shape[0] == gpop_ss)
            # assert(pool.shape[0] == pool_ss)
            # assert(rpop.shape[0] == rpop_ss)

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

            # Estimate CN by eps-contamination
            bn_min=gum.BayesNet(bn)
            bn_max=gum.BayesNet(bn)
            for n in bn.nodes():
                x = (1-eps)*bn_min.cpt(n)
                bn_min.cpt(n).fillWith(x)
                bn_max.cpt(n).fillWith(x+eps)

            # Create CN
            cn = gum.CredalNet(bn_min,bn_max)
            cn.intervalToCredal()

            # Extract random subset within simplex
            bns_sample = get_simplex_inner(cn, n_bns)

            # Debug
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
                t = np.quantile(L_im, e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred = llr_gpop < t

                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred) / tp
                power_bn = power_bn + [tpr]

            # Store BN results for this data sample
            results_ds["power_BN"] = power_bn

            # MIA (CN)
            best_sum = 0
            best_bn_idx = 0
            for i, bn_s in enumerate(tqdm(bns_sample, desc="Credal samples", unit="item", dynamic_ncols=True, leave=False)):

                # Estimate the distribution of LLR(x) from rpop (i..e under H_0)
                bn_s_ie = gum.LazyPropagation(bn_s)
                L_im_s = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, bn_s_ie), axis=1).dropna()

                L_im_s_sum = np.sum(L_im_s)
                if L_im_s_sum > best_sum: 
                    best_sum = L_im_s_sum
                    best_bn_idx = i

            # Get best BN inside simplex
            best_bn_s_ie = gum.LazyPropagation(bns_sample[best_bn_idx])
            L_im_best = rpop.apply(lambda x: LLR(x.to_dict(), bn_theta_ie, best_bn_s_ie), axis=1).dropna()

            # Compute LLR(x) on general population
            llr_gpop_s = gpop[[*bn.names()]].apply(lambda x: LLR(x.to_dict(), bn_theta_ie, best_bn_s_ie), axis=1)

            # Init the power vector (BN vertex)
            power_bn_s = []

            # For each error ...
            for e in error:

                # Compute threshold
                t = np.quantile(L_im_best, e).item()

                # LLR test. Reject H_0? True => x in pool; False => x in rpop.
                y_pred_s = llr_gpop_s < t
                
                # Compute and store power (tpr)
                tpr = sum(gpop["in-pool"] & y_pred_s) / tp
                power_bn_s = power_bn_s + [tpr]
            
            # Store CN results for this data sample
            results_ds["power_CN"] = power_bn_s

            # Add ds results to final results
            results["power_BN"] += results_ds["power_BN"]
            results["power_CN"] += results_ds["power_CN"]
            # results_ds.drop(["error", "power_BN"], axis=1, inplace=True)

            # results["CN_avg"] += results_ds.mean(axis=1)
            # results["CN_min"] += results_ds.min(axis=1)
            # results["CN_max"] += results_ds.max(axis=1)

            # Debug
            # assert(results_ds.shape[1] == n_bns)
            # assert(results.shape[1] == 6)
        except:
            with open("./results/cont/log.txt", "a") as log: log.write(f"{conf['meta']}: error with sample {ds}.\n")

                # Debug
                # log.write(traceback.format_exc())

    # Compute average results and save them
    results[["power_BN", "power_CN"]] /= n_ds
    results.to_csv(f"./results/cont/{conf['meta']}-compl{compl}.csv", index=False)
    with open("./results/cont/log.txt", "a") as log: log.write(f"{conf['meta']}: [OK]\n")