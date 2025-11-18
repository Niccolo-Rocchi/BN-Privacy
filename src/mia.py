import math
import sys

import numpy as np
import pandas as pd
import pyagrum as gum
from scipy.stats import norm
from sklearn import metrics
from scipy.optimize import minimize

from src.config import get_cur_dir, set_seed
from src.defense import noisy_bn


# MIA attack vs a BN
def mia_vs_bn(exp, config) -> dict:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Init results
    power_res = pd.DataFrame({"error": eval(config["error"])})
    auc_res = pd.DataFrame({"sample": range(config["samples"])})
    auc_res["exp"] = exp

    # Read data
    gpop = pd.read_csv(f'{cur_dir / config["data_path"]}/{exp}.csv')

    # Set seed
    set_seed()

    # For each data sample ...
    auc_bns_dict = dict()
    for sample in range(config["samples"]):

        # ... read the BNs as estimated from rpop and pool, ...
        bn_theta = gum.loadBN(
            f"{cur_dir}/{config['bns_path']}/rpop/bn_{exp}_sample{sample}.bif"
        )
        bn_theta_hat = gum.loadBN(
            f"{cur_dir}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif"
        )

        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        bn_theta_ie = gum.LazyPropagation(bn_theta)

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_theta.nodes())]

        # try:

        # ... and perform membership inference on gpop
        power_vec, auc = run_mia(
            bn_theta_hat_ie,
            bn_theta_ie,
            rpop,
            gpop,
            gpop[f"in-pool-{sample}"],
            eval(config["error"]),
        )
        power_res[f"power_BN_sample{sample}"] = power_vec
        auc_bns_dict[sample] = auc

        # except Exception:

        #     # Debug
        #     with open(f"{results_path}/log.txt", "a") as log:
        #         log.write(f"{exp}: error with sample {sample} (BN).\n")
        #         log.write(traceback.format_exc())

    # Save results
    power_res.to_csv(f'{cur_dir}/{config["results_path"]}/bns/power_bn_{exp}.csv', index=False)
    
    # Return
    auc_res["auc_bn"] = auc_res.apply(lambda row: auc_bns_dict[row["sample"]], axis=1)

    return auc_res


# MIA attack vs a CN
def mia_vs_cn(exp, config) -> pd.DataFrame:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Init results
    power_res = pd.DataFrame({"error": eval(config["error"])})
    auc_res = pd.DataFrame({"sample": range(config["samples"])})
    auc_res["exp"] = exp

    # Read data
    gpop = pd.read_csv(f'{cur_dir / config["data_path"]}/{exp}.csv')

    # Set seed
    set_seed()

    # For each data sample ...
    auc_cns_dict = dict()
    for sample in range(config["samples"]):

        # ... read the BN as inferred from the CN
        bn_theta_hat = gum.loadBN(
            f'{cur_dir}/{config["atk_path"]}/bn_{exp}_sample{sample}.bif'
        )

        # ... read the BN as estimated from rpop, ...
        bn_theta = gum.loadBN(
            f"{cur_dir}/{config['bns_path']}/rpop/bn_{exp}_sample{sample}.bif"
        )

        bn_theta_hat_ie = gum.LazyPropagation(bn_theta_hat)
        bn_theta_ie = gum.LazyPropagation(bn_theta)

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_theta.nodes())]

        # try:

        # ... and perform membership inference on gpop
        power_vec, auc = run_mia(
            bn_theta_hat_ie,
            bn_theta_ie,
            rpop,
            gpop,
            gpop[f"in-pool-{sample}"],
            eval(config["error"]),
        )
        power_res[f"power_CN_sample{sample}"] = power_vec
        auc_cns_dict[sample] = auc

        # except Exception:

        #     # Debug
        #     with open(f"{results_path}/log.txt", "a") as log:
        #         log.write(f"{exp}: error with sample {sample} (BN).\n")
        #         log.write(traceback.format_exc())

    # Save results
    power_res.to_csv(
        f'{cur_dir}/{config["results_path"]}/cns/power_cn_{exp}.csv',
        index=False)

    # Return
    auc_res["auc_cn"] = auc_res.apply(lambda row: auc_cns_dict[row["sample"]], axis=1)

    return auc_res


# Get theoretical power
def theoretical_power(exp, config) -> None:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Read data
    bn = gum.loadBN(f'{get_cur_dir(config) / config["bns_path"]}/gt/{exp}.bif')
    results = pd.read_csv(f'{cur_dir}/{config["results_path"]}/bns/power_bn_{exp}.csv')

    # Set seed
    set_seed()

    # Compute bound
    bound = math.sqrt(bn.dim() / int(config["gpop_ss"] * config["pool_prop"]))

    # Find power (beta) for any error (alpha) given theoretical bound
    z_alpha = [norm.ppf(1 - i).item() for i in eval(config["error"])]
    z_one_minus_beta = [bound - i for i in z_alpha]
    beta = [norm.cdf(i).item() for i in z_one_minus_beta]

    # Save results
    results["power_bound"] = beta
    results.to_csv(f'{cur_dir}/{config["results_path"]}/bns/power_bn_{exp}.csv', index=False)

    return

# Find eps s.t. |AUC(eps) - AUC(CN)| < tol
def find_epsilon(exp, config) -> dict:

    # Get current directory
    cur_dir = get_cur_dir(config)

    # Init results
    power_res = pd.DataFrame({"error": eval(config["error"])})

    # Read data
    gpop = pd.read_csv(f'{cur_dir / config["data_path"]}/{exp}.csv')
    gpop_ss = config["gpop_ss"]
    pool_ss = int(gpop_ss * config["pool_prop"])
    auc_res = pd.read_csv(f'{cur_dir}/{config["auc_meta"]}')
    auc_res = auc_res[auc_res["exp"] == exp]
    eps_vec = eval(config["eps_vec"])

    # Set seed
    set_seed()

    # For each data sample ...
    eps_dict = dict()
    auc_noisy_dict = dict()
    for sample in range(config["samples"]):

        # ... read the BNs as estimated from rpop and pool, ...
        bn_theta = gum.loadBN(
            f"{cur_dir}/{config['bns_path']}/rpop/bn_{exp}_sample{sample}.bif"
        )
        bn_theta_hat = gum.loadBN(
            f"{cur_dir}/{config['bns_path']}/pool/bn_{exp}_sample{sample}.bif"
        )

        # ... retrieve rpop, ...
        rpop = gpop[gpop[f"in-rpop-{sample}"]].iloc[:, : len(bn_theta.nodes())]

        # ... get CN AUC, ...
        auc_cn = auc_res.loc[auc_res["sample"] == sample, "auc_cn"].values[0]

        # ... init results, ...
        eps_dict[sample] = None
        auc_noisy_dict[sample] = None

        # ... and find epsilon
        for eps in eps_vec:
                
            # Get noisy BN
            scale = (2 * bn_theta_hat.size()) / (pool_ss * eps)
            bn_noisy = noisy_bn(bn_theta_hat, scale)

            bn_noisy_ie = gum.LazyPropagation(bn_noisy)
            bn_theta_ie = gum.LazyPropagation(bn_theta)

            # Perform membership inference on gpop
            power_vec, auc = run_mia(
                bn_noisy_ie,
                bn_theta_ie,
                rpop,
                gpop,
                gpop[f"in-pool-{sample}"],
                eval(config["error"]),
            )

            # Condition on |AUC(eps) - AUC(CN)|
            if abs(auc_cn - auc) < config["tol"]:
                eps_dict[sample] = eps
                auc_noisy_dict[sample] = auc
                power_res[f"power_BN_noisy_sample{sample}"] = power_vec
                break
 
    # Save results
    power_res.to_csv(
        f'{cur_dir}/{config["results_path"]}/bn_noisy/power_bn_{exp}.csv',
        index=False)
    
    # Return
    auc_res["epsilon"] = auc_res.apply(lambda row: eps_dict[row["sample"]], axis=1)
    auc_res["auc_noisy_bn"] = auc_res.apply(lambda row: auc_noisy_dict[row["sample"]], axis=1)

    return auc_res


# MIA: membership inference attack
def run_mia(model, baseline, rpop, gpop, ground_truth, error_vec):

    # Compute llr(x) on reference and general populations
    llr_ref = (
        rpop.apply(lambda x: get_llr(x.to_dict(), baseline, model), axis=1)
        .dropna()
        .sort_values()
    )
    llr_gen = gpop[[*rpop.columns]].apply(
        lambda x: get_llr(x.to_dict(), baseline, model), axis=1
    )

    power_vec = []

    # Get the power for each error
    for error in error_vec:
        power = get_power(llr_ref, llr_gen, ground_truth, error)
        power_vec.append(power)

    # Compute and store AUC
    auc = metrics.auc(error_vec, power_vec)

    return power_vec, auc


# Get the attack power related to a fixed error
def get_power(llr_ref, llr_gen, ground_truth, error) -> float:

    # Compute the threshold
    t = np.quantile(llr_ref, 1 - error).item()

    # Test: L(x) > t => reject H_0 => assign `x` to target_pop
    y_pred = llr_gen > t

    # Compute power (i.e., true positive rate)
    power = sum(ground_truth & y_pred) / sum(ground_truth)

    return power


# Log-likelihood function
def get_ll(x: dict, theta):

    # Erase all evidences and apply addEvidence(key,value) for every pairs in x
    theta.setEvidence(x)

    # Compute P(x | theta)
    ll = theta.evidenceProbability()

    return np.log(ll)


# Log-likelihood ratio (llr) function
def get_llr(x: dict, theta, theta_hat):

    # Compute log-likelihoods
    ll_theta = get_ll(x, theta)
    ll_theta_hat = get_ll(x, theta_hat)

    return ll_theta_hat - ll_theta
